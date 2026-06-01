#include <obs-module.h>
#include "plugin-support.h"
#include <onnxruntime_cxx_api.h>
// DML provider loaded dynamically at runtime via GetProcAddress
#define NOMINMAX
#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <dxgi1_6.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")

// ----------------------------------------------------------
// SAFETY LIMITS
// ----------------------------------------------------------
#define MAX_WIDTH  3840
#define MAX_HEIGHT 2160
#define MIN_WIDTH  64
#define MIN_HEIGHT 64

// Settings property keys
#define PROP_SMOOTHING    "smoothing"
#define PROP_THRESHOLD    "threshold"
#define PROP_FEATHER      "feather"
#define PROP_EROSION      "erosion"
#define PROP_EDGE_BLUR    "edge_blur"
#define PROP_INFER_SCALE  "infer_scale"
#define PROP_RESPONSIVENESS "responsiveness"
#define PROP_DEVICE       "inference_device"

// ----------------------------------------------------------
// Filter data
// ----------------------------------------------------------
struct bg_filter_data {
	obs_source_t    *source;

	// ONNX Runtime
	Ort::Env        *env;
	Ort::Session    *session;
	Ort::MemoryInfo *memory_info;

	// RVM recurrent states
	std::vector<float>   r1, r2, r3, r4;
	std::vector<int64_t> r1_shape, r2_shape, r3_shape, r4_shape;

	// GPU resources (render thread only)
	gs_texrender_t  *texrender;
	gs_texrender_t  *small_texrender;  // downscaled for inference
	gs_stagesurf_t  *staging;          // small staging surface
	gs_texture_t    *mask_tex;
	gs_effect_t     *mask_effect;
	gs_samplerstate_t *linear_sampler;  // bilinear filter for mask upscale

	// CPU frame/mask buffers shared between threads
	std::vector<uint8_t> frame_bgra;
	uint32_t frame_infer_w;  // actual width of staged frame
	uint32_t frame_infer_h;  // actual height of staged frame
	std::vector<float>   prev_mask;   // M6: previous alpha for smoothing
	std::vector<uint8_t> mask_rgba;   // written by inference thread
	uint32_t mask_w;  // dimensions of mask_rgba (inference res)
	uint32_t mask_h;

	// Thread synchronisation
	std::thread              infer_thread;
	std::mutex               frame_mutex;
	std::condition_variable  frame_cv;
	std::mutex               mask_mutex;
	std::atomic<bool>        frame_ready;
	std::atomic<bool>        has_mask;
	std::atomic<bool>        running;
	std::atomic<bool>        infer_busy;

	uint32_t width;
	uint32_t height;
	uint32_t frame_linesize;

	bool initialized;
	bool failed;
	bool using_cuda;
	int  frame_count;
	int  error_count;

	// M7: user settings
	std::mutex settings_mutex;
	float      smoothing;
	float      threshold;
	float      feather;
	int        erosion;
	int        edge_blur;
	float      infer_scale;
	float      responsiveness;
	uint32_t   prev_infer_w;  // detect dimension changes to reset states
	uint32_t   prev_infer_h;
	int        device_id;        // desired DML adapter index (-1 = CPU)
	int        active_device_id; // device the live session is bound to
};

// ----------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------
static const char        *bg_filter_name(void *unused);
static void              *bg_filter_create(obs_data_t *settings, obs_source_t *source);
static void               bg_filter_destroy(void *data);
static void               bg_filter_render(void *data, gs_effect_t *effect);
static void               bg_filter_update(void *data, obs_data_t *settings);
static obs_properties_t  *bg_filter_properties(void *data);
static void               bg_filter_defaults(obs_data_t *settings);
static bool               init_onnx(bg_filter_data *f, uint32_t w, uint32_t h);
static void               inference_thread_func(bg_filter_data *f);

// ----------------------------------------------------------
// Helpers
// ----------------------------------------------------------
static bool validate_dims(uint32_t w, uint32_t h)
{
	return w >= MIN_WIDTH && h >= MIN_HEIGHT &&
	       w <= MAX_WIDTH && h <= MAX_HEIGHT;
}

// ----------------------------------------------------------
// DirectML adapter enumeration
//
// The DML execution provider picks a GPU by an integer device id that
// matches the DXGI adapter enumeration order. We enumerate the same way
// here so the dropdown index lines up with what DML expects, and so we
// can show real adapter names instead of opaque numbers.
// ----------------------------------------------------------
struct dml_adapter_info {
	int         index;          // DXGI index == DML device id
	std::string name;
	uint32_t    vendor_id;
	uint64_t    dedicated_vram;
};

static std::vector<dml_adapter_info> enumerate_dml_adapters()
{
	std::vector<dml_adapter_info> out;

	Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
		return out;

	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	for (UINT i = 0;
	     factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf())
		     != DXGI_ERROR_NOT_FOUND;
	     i++) {
		DXGI_ADAPTER_DESC1 desc;
		if (FAILED(adapter->GetDesc1(&desc)))
			continue;
		// Skip the Microsoft Basic Render Driver (software fallback)
		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			continue;

		char namebuf[256] = {0};
		WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
				    namebuf, sizeof(namebuf) - 1,
				    nullptr, nullptr);

		dml_adapter_info info;
		info.index          = (int)i;
		info.name           = namebuf;
		info.vendor_id      = desc.VendorId;
		info.dedicated_vram = desc.DedicatedVideoMemory;
		out.push_back(std::move(info));
	}
	return out;
}

// ----------------------------------------------------------
// Tear down the ONNX session + inference thread so the caller can
// re-init. Used on resolution change and on inference-device change.
// Must be called from the render thread.
// ----------------------------------------------------------
static void teardown_inference(bg_filter_data *f)
{
	f->running.store(false);
	f->frame_cv.notify_all();
	if (f->infer_thread.joinable())
		f->infer_thread.join();
	f->initialized = false;
	f->has_mask.store(false);
	f->prev_mask.clear();
	f->r1.clear(); f->r1_shape = {1,1,1,1};
	f->r2.clear(); f->r2_shape = {1,1,1,1};
	f->r3.clear(); f->r3_shape = {1,1,1,1};
	f->r4.clear(); f->r4_shape = {1,1,1,1};
	delete f->session;     f->session     = nullptr;
	delete f->memory_info; f->memory_info = nullptr;
	delete f->env;         f->env         = nullptr;
}

// ----------------------------------------------------------
// Background inference thread
// ----------------------------------------------------------
static void inference_thread_func(bg_filter_data *f)
{
	obs_log(LOG_INFO, "BG Removal: Inference thread started");

	while (f->running.load()) {
		{
			std::unique_lock<std::mutex> lock(f->frame_mutex);
			f->frame_cv.wait_for(lock,
				std::chrono::milliseconds(100),
				[f]{ return f->frame_ready.load()
				          || !f->running.load(); });
			if (!f->running.load()) break;
			if (!f->frame_ready.load()) continue;
			f->frame_ready.store(false);
		}

		if (!f->session || !f->memory_info) continue;
		f->infer_busy.store(true);

		uint32_t W = f->width;
		uint32_t H = f->height;

		float smoothing, threshold, feather, responsiveness;
		int erosion, edge_blur;
		{
			std::lock_guard<std::mutex> lock(f->settings_mutex);
			smoothing      = f->smoothing;
			threshold      = f->threshold;
			feather        = f->feather;
			erosion        = f->erosion;
			edge_blur      = f->edge_blur;
			responsiveness = f->responsiveness;
		}

		std::vector<uint8_t> local_frame;
		uint32_t local_linesize = 0;
		uint32_t IW, IH;
		{
			std::lock_guard<std::mutex> lock(f->frame_mutex);
			local_frame    = f->frame_bgra;
			local_linesize = f->frame_linesize;
			IW = f->frame_infer_w;
			IH = f->frame_infer_h;
		}
		if (local_frame.empty() || !IW || !IH) {
			f->infer_busy.store(false); continue;
		}

		// Detect dimension change: reset recurrent states + prev_mask
		if (IW != f->prev_infer_w || IH != f->prev_infer_h) {
			obs_log(LOG_INFO,
				"BG Removal: Infer size changed %dx%d -> %dx%d, resetting states",
				f->prev_infer_w, f->prev_infer_h, IW, IH);
			f->r1 = {0.0f}; f->r1_shape = {1,1,1,1};
			f->r2 = {0.0f}; f->r2_shape = {1,1,1,1};
			f->r3 = {0.0f}; f->r3_shape = {1,1,1,1};
			f->r4 = {0.0f}; f->r4_shape = {1,1,1,1};
			f->prev_mask.clear();
			f->prev_infer_w = IW;
			f->prev_infer_h = IH;
		}

		uint32_t PIW = ((IW + 15) / 16) * 16;
		uint32_t PIH = ((IH + 15) / 16) * 16;
		size_t   infer_pixels = (size_t)PIW * PIH;

		try {
			auto t_start = std::chrono::steady_clock::now();

			// Frame already at inference resolution from GPU downscale.
			// Direct BGRA -> float RGB channel-first with padding.
			std::vector<float> rgb(3 * infer_pixels, 0.0f);
			float *R = rgb.data();
			float *G = rgb.data() + infer_pixels;
			float *B = rgb.data() + infer_pixels * 2;

			for (uint32_t y = 0; y < IH; y++) {
				const uint8_t *row =
					local_frame.data() +
					(size_t)y * local_linesize;
				for (uint32_t x = 0; x < IW; x++) {
					size_t i = (size_t)y * PIW + x;
					B[i] = row[x*4+0] / 255.0f;
					G[i] = row[x*4+1] / 255.0f;
					R[i] = row[x*4+2] / 255.0f;
				}
			}

			std::vector<int64_t> src_shape = {1,3,(int64_t)PIH,(int64_t)PIW};

			// downsample_ratio must be 1.0 - other values crash DirectML
			float ratio_val = 1.0f;
			std::vector<int64_t> ratio_shape = {1};

			// Responsiveness: decay recurrent states to reduce temporal memory.
			// 1.0 = full decay (model treats each frame independently)
			// 0.0 = no decay (full temporal memory, smoothest but laggiest)
			if (responsiveness > 0.001f) {
				float keep = 1.0f - responsiveness;
				for (auto &v : f->r1) v *= keep;
				for (auto &v : f->r2) v *= keep;
				for (auto &v : f->r3) v *= keep;
				for (auto &v : f->r4) v *= keep;
			}

			auto mkt = [&](std::vector<float> &buf,
				       std::vector<int64_t> &shape) -> Ort::Value {
				return Ort::Value::CreateTensor<float>(
					*f->memory_info,
					buf.data(), buf.size(),
					shape.data(), shape.size());
			};

			std::vector<Ort::Value> inputs;
			inputs.reserve(6);
			inputs.push_back(mkt(rgb,   src_shape));
			inputs.push_back(mkt(f->r1, f->r1_shape));
			inputs.push_back(mkt(f->r2, f->r2_shape));
			inputs.push_back(mkt(f->r3, f->r3_shape));
			inputs.push_back(mkt(f->r4, f->r4_shape));
			inputs.push_back(Ort::Value::CreateTensor<float>(
				*f->memory_info, &ratio_val, 1,
				ratio_shape.data(), ratio_shape.size()));

			const char *inames[] = {"src","r1i","r2i","r3i","r4i",
						"downsample_ratio"};
			const char *onames[] = {"fgr","pha","r1o","r2o","r3o","r4o"};

			auto t_pre = std::chrono::steady_clock::now();

			auto out = f->session->Run(
				Ort::RunOptions{nullptr},
				inames, inputs.data(), 6,
				onames, 6);

			auto t_infer = std::chrono::steady_clock::now();

			if (out.size() != 6) {
				obs_log(LOG_ERROR, "BG Removal: Expected 6 outputs got %zu",
					out.size());
				f->infer_busy.store(false);
				continue;
			}

			size_t pha_n = out[1].GetTensorTypeAndShapeInfo().GetElementCount();
			if (pha_n != infer_pixels) {
				obs_log(LOG_ERROR,
					"BG Removal: Alpha size mismatch expected=%zu got=%zu",
					infer_pixels, pha_n);
				f->infer_busy.store(false);
				continue;
			}

			// Update recurrent states from model outputs
			auto upd = [](Ort::Value &t,
				      std::vector<float> &s,
				      std::vector<int64_t> &sh) {
				auto info = t.GetTensorTypeAndShapeInfo();
				sh = info.GetShape();
				size_t n = info.GetElementCount();
				float *p = t.GetTensorMutableData<float>();
				s.assign(p, p + n);
			};
			upd(out[2], f->r1, f->r1_shape);
			upd(out[3], f->r2, f->r2_shape);
			upd(out[4], f->r3, f->r3_shape);
			upd(out[5], f->r4, f->r4_shape);

			float *pha = out[1].GetTensorMutableData<float>();

			size_t infer_unpadded = (size_t)IW * IH;

			// M6: init prev_mask at inference resolution
			if (f->prev_mask.size() != infer_unpadded)
				f->prev_mask.assign(infer_unpadded, 0.0f);

			// --- Process mask entirely at inference resolution ---
			// Step 1: Smoothing + threshold + feather at IW×IH
			float feather_lo = threshold;
			float feather_hi = threshold + feather;
			std::vector<uint8_t> small_mask(infer_unpadded);

			for (uint32_t y = 0; y < IH; y++) {
				for (uint32_t x = 0; x < IW; x++) {
					size_t src_i = (size_t)y * PIW + x;
					size_t dst_i = (size_t)y * IW  + x;

					float raw = std::clamp(pha[src_i], 0.0f, 1.0f);

					float blended = f->prev_mask[dst_i] * smoothing
					              + raw * (1.0f - smoothing);
					f->prev_mask[dst_i] = blended;

					float final_a;
					if (feather > 0.001f) {
						float t = (blended - feather_lo) /
							  (feather_hi - feather_lo + 0.0001f);
						final_a = std::clamp(t, 0.0f, 1.0f);
					} else {
						final_a = (blended < threshold) ? 0.0f : blended;
					}

					small_mask[dst_i] = (uint8_t)(final_a * 255.0f);
				}
			}

			// Step 2: Erosion at inference resolution (much faster)
			if (erosion > 0) {
				std::vector<uint8_t> eroded(infer_unpadded);
				for (uint32_t y = 0; y < IH; y++) {
					for (uint32_t x = 0; x < IW; x++) {
						uint8_t min_a = 255;
						int y0 = std::max((int)y - erosion, 0);
						int y1 = std::min((int)y + erosion, (int)IH - 1);
						int x0 = std::max((int)x - erosion, 0);
						int x1 = std::min((int)x + erosion, (int)IW - 1);
						for (int ey = y0; ey <= y1; ey++)
							for (int ex = x0; ex <= x1; ex++)
								min_a = std::min(min_a,
									small_mask[(size_t)ey*IW+ex]);
						eroded[(size_t)y * IW + x] = min_a;
					}
				}
				small_mask = std::move(eroded);
			}

			// Step 3: Edge blur at inference resolution
			if (edge_blur > 0) {
				std::vector<uint8_t> blurred(infer_unpadded);
				for (uint32_t y = 0; y < IH; y++) {
					for (uint32_t x = 0; x < IW; x++) {
						int sum = 0, count = 0;
						int y0 = std::max((int)y - edge_blur, 0);
						int y1 = std::min((int)y + edge_blur, (int)IH - 1);
						int x0 = std::max((int)x - edge_blur, 0);
						int x1 = std::min((int)x + edge_blur, (int)IW - 1);
						for (int by = y0; by <= y1; by++)
							for (int bx = x0; bx <= x1; bx++) {
								sum += small_mask[(size_t)by*IW+bx];
								count++;
							}
						blurred[(size_t)y * IW + x] = (uint8_t)(sum / count);
					}
				}
				small_mask = std::move(blurred);
			}

			// Step 4: Write small mask as RGBA (GPU does the upscale)
			std::vector<uint8_t> new_mask(infer_unpadded * 4);
			for (size_t i = 0; i < infer_unpadded; i++) {
				uint8_t a = small_mask[i];
				new_mask[i*4+0] = a;
				new_mask[i*4+1] = a;
				new_mask[i*4+2] = a;
				new_mask[i*4+3] = 255;
			}

			{
				std::lock_guard<std::mutex> lock(f->mask_mutex);
				f->mask_rgba = std::move(new_mask);
				f->mask_w = IW;
				f->mask_h = IH;
			}
			f->has_mask.store(true);
			f->error_count = 0;
			f->frame_count++;

			auto t_end = std::chrono::steady_clock::now();

			if (f->frame_count % 300 == 0) {
				auto ms = [](auto a, auto b) {
					return std::chrono::duration<float, std::milli>(b - a).count();
				};
				float mn = pha[0], mx = pha[0];
				for (uint32_t y = 0; y < IH; y++)
					for (uint32_t x = 0; x < IW; x++) {
						float v = pha[(size_t)y*PIW+x];
						mn = std::min(mn, v);
						mx = std::max(mx, v);
					}
				obs_log(LOG_INFO,
					"BG Removal: Frame %d | %s %dx%d | "
					"prep=%.1fms infer=%.1fms mask=%.1fms total=%.1fms (%.0ffps)",
					f->frame_count,
					f->using_cuda ? "DirectML" : "CPU",
					IW, IH,
					ms(t_start, t_pre),
					ms(t_pre, t_infer),
					ms(t_infer, t_end),
					ms(t_start, t_end),
					1000.0f / ms(t_start, t_end));
			}

		} catch (const Ort::Exception &e) {
			f->error_count++;
			obs_log(LOG_ERROR, "BG Removal: ONNX error [%d]: %s (x%d)",
				e.GetOrtErrorCode(), e.what(), f->error_count);
			if (f->error_count >= 10) {
				obs_log(LOG_ERROR, "BG Removal: Too many errors.");
				f->failed = true;
				f->running.store(false);
			}
		} catch (const std::exception &e) {
			f->error_count++;
			obs_log(LOG_ERROR, "BG Removal: Error: %s", e.what());
		} catch (...) {
			f->error_count++;
			obs_log(LOG_ERROR, "BG Removal: Unknown error (x%d)", f->error_count);
		}

		f->infer_busy.store(false);
	}

	obs_log(LOG_INFO, "BG Removal: Inference thread stopped");
}

// ----------------------------------------------------------
// Initialize ONNX
// ----------------------------------------------------------
static bool init_onnx(bg_filter_data *f, uint32_t W, uint32_t H)
{
	obs_log(LOG_INFO, "BG Removal: Initializing ONNX for %dx%d...", W, H);

	if (!validate_dims(W, H)) {
		obs_log(LOG_ERROR, "BG Removal: Invalid dims %dx%d", W, H);
		return false;
	}

	uint32_t PW = ((W + 15) / 16) * 16;
	uint32_t PH = ((H + 15) / 16) * 16;
	if (PW != W || PH != H)
		obs_log(LOG_INFO, "BG Removal: Padded %dx%d -> %dx%d", W, H, PW, PH);

	try {
		f->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "bg-removal");
		obs_log(LOG_INFO, "BG Removal: ONNX Env OK");

		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(2);
		opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

		// DirectML: uses DirectX 12, works on RTX 5080 Blackwell.
		// NOTE: CUDA hard-crashes on RTX 5080 (SM 12.0) - ONNX Runtime 1.24.3
		// CUDA provider does not yet support Blackwell architecture (March 2026).
		// Load DML function dynamically - it lives in onnxruntime.dll at runtime
		// but is not exported from onnxruntime.lib, so we use GetProcAddress.
		typedef OrtStatus *(ORT_API_CALL *DML_FN)(OrtSessionOptions *, int);
		f->using_cuda = false;

		// Which adapter did the user pick? (-1 = force CPU)
		int dev;
		{
			std::lock_guard<std::mutex> lock(f->settings_mutex);
			dev = f->device_id;
		}
		f->active_device_id = dev;

		// Resolve a readable name for logging / verification.
		std::string dev_name = "CPU";
		if (dev >= 0) {
			for (const auto &a : enumerate_dml_adapters())
				if (a.index == dev) dev_name = a.name;
		}

		if (dev < 0) {
			obs_log(LOG_INFO,
				"BG Removal: Inference device = CPU (user-selected)");
		} else {
			HMODULE hOrt = GetModuleHandleA("onnxruntime.dll");
			if (hOrt) {
				DML_FN dml_fn = (DML_FN)GetProcAddress(
					hOrt, "OrtSessionOptionsAppendExecutionProvider_DML");
				if (dml_fn) {
					OrtStatus *dml_st = dml_fn(opts, dev);
					if (dml_st == nullptr) {
						f->using_cuda = true;
						obs_log(LOG_INFO,
							"BG Removal: DirectML provider OK on device %d (%s)",
							dev, dev_name.c_str());
					} else {
						obs_log(LOG_WARNING,
							"BG Removal: DirectML device %d (%s) error, using CPU",
							dev, dev_name.c_str());
						Ort::GetApi().ReleaseStatus(dml_st);
					}
				} else {
					obs_log(LOG_WARNING,
						"BG Removal: DML function not found in onnxruntime.dll, using CPU");
				}
			} else {
				obs_log(LOG_WARNING,
					"BG Removal: onnxruntime.dll not loaded yet, using CPU");
			}
		}

		char *mpath = obs_module_file("models/rvm_mobilenetv3_fp32.onnx");
		if (!mpath) {
			obs_log(LOG_ERROR, "BG Removal: Model not found!");
			return false;
		}
		std::string  ps(mpath);
		std::wstring wp(ps.begin(), ps.end());
		obs_log(LOG_INFO, "BG Removal: Model: %s", ps.c_str());
		bfree(mpath);

		f->session = new Ort::Session(*f->env, wp.c_str(), opts);
		obs_log(LOG_INFO, "BG Removal: Model loaded OK");

		if (f->session->GetInputCount() != 6 ||
		    f->session->GetOutputCount() != 6) {
			obs_log(LOG_ERROR, "BG Removal: Wrong model format!");
			delete f->session;     f->session     = nullptr;
			delete f->env;         f->env         = nullptr;
			return false;
		}

		f->memory_info = new Ort::MemoryInfo(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

		// Start states as {1,1,1,1} - model bootstraps itself on first frame
		f->r1 = {0.0f}; f->r1_shape = {1,1,1,1};
		f->r2 = {0.0f}; f->r2_shape = {1,1,1,1};
		f->r3 = {0.0f}; f->r3_shape = {1,1,1,1};
		f->r4 = {0.0f}; f->r4_shape = {1,1,1,1};
		obs_log(LOG_INFO, "BG Removal: States initialized as 1x1x1x1");

		f->width       = W;
		f->height      = H;
		f->initialized = true;

		f->running.store(true);
		f->infer_thread = std::thread(inference_thread_func, f);

		obs_log(LOG_INFO, "BG Removal: *** READY! Backend=%s %dx%d ***",
			f->using_cuda ? "DirectML(GPU)" : "CPU", W, H);
		return true;

	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "BG Removal: ONNX init error [%d]: %s",
			e.GetOrtErrorCode(), e.what());
	} catch (const std::bad_alloc &) {
		obs_log(LOG_ERROR, "BG Removal: Out of memory!");
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "BG Removal: Init error: %s", e.what());
	} catch (...) {
		obs_log(LOG_ERROR, "BG Removal: Unknown init error!");
	}
	return false;
}

// ----------------------------------------------------------
// Filter name
// ----------------------------------------------------------
static const char *bg_filter_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "BG Removal";
}

// ----------------------------------------------------------
// M7: Settings UI
// ----------------------------------------------------------
static obs_properties_t *bg_filter_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	obs_property_t *p;

	// --- Performance ---
	obs_properties_add_text(props, "info_perf",
		"<b>Performance</b>", OBS_TEXT_INFO);

	// Inference device picker (multi-GPU offload)
	p = obs_properties_add_list(props, PROP_DEVICE,
		"Inference Device",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	for (const auto &a : enumerate_dml_adapters()) {
		char label[320];
		double gb = (double)a.dedicated_vram /
			    (1024.0 * 1024.0 * 1024.0);
		snprintf(label, sizeof(label), "%s (%.1f GB)",
			 a.name.c_str(), gb);
		obs_property_list_add_int(p, label, a.index);
	}
	obs_property_list_add_int(p, "CPU (no GPU acceleration)", -1);
	obs_property_set_long_description(p,
		"Which GPU runs the AI inference. On a multi-GPU system you can "
		"offload this to a secondary card (e.g. an Intel Arc) to free up "
		"your main GPU for rendering and encoding. Changing this rebuilds "
		"the model session.");

	p = obs_properties_add_float_slider(props, PROP_INFER_SCALE,
		"Inference Scale", 0.25, 1.0, 0.05);
	obs_property_set_long_description(p,
		"Scales the frame before AI inference, then upscales the mask on GPU. "
		"Lower = faster tracking, higher = sharper mask edges. "
		"0.25 is optimal for 60fps on most GPUs.");

	obs_properties_add_text(props, "info_scale",
		"Controls how fast the mask tracks your movement. "
		"0.25 = fastest (recommended)  |  0.50 = sharper edges but slower",
		OBS_TEXT_INFO);

	p = obs_properties_add_float_slider(props, PROP_RESPONSIVENESS,
		"Responsiveness", 0.0, 1.0, 0.05);
	obs_property_set_long_description(p,
		"Decays the AI model's temporal memory each frame. "
		"Higher = faster reaction to movement, but more edge flicker.");

	obs_properties_add_text(props, "info_responsive",
		"Fixes the mask lagging behind fast movement. "
		"0 = smoothest (laggy)  |  0.3 = balanced  |  0.7+ = snappy but jittery",
		OBS_TEXT_INFO);

	// --- Mask Quality ---
	obs_properties_add_text(props, "info_quality",
		"<b>Mask Quality</b>", OBS_TEXT_INFO);

	p = obs_properties_add_float_slider(props, PROP_THRESHOLD,
		"Alpha Threshold", 0.0, 0.5, 0.01);
	obs_property_set_long_description(p,
		"Pixels with alpha below this value are cut to fully transparent.");

	obs_properties_add_text(props, "info_threshold",
		"Removes faint background ghosts. Increase if you see "
		"semi-transparent areas in the background.",
		OBS_TEXT_INFO);

	p = obs_properties_add_float_slider(props, PROP_FEATHER,
		"Edge Feather", 0.0, 0.3, 0.01);
	obs_property_set_long_description(p,
		"Softens the transition at edges above the threshold.");

	obs_properties_add_text(props, "info_feather",
		"Creates a soft blend at the cutoff edge instead of a hard line. "
		"0 = hard cut  |  0.05 = subtle  |  0.15 = very soft",
		OBS_TEXT_INFO);

	p = obs_properties_add_int_slider(props, PROP_EROSION,
		"Mask Erosion", 0, 10, 1);
	obs_property_set_long_description(p,
		"Shrinks the mask boundary inward by N pixels.");

	obs_properties_add_text(props, "info_erosion",
		"Eats the bright outline/halo around your silhouette. "
		"1-2 = subtle  |  3-5 = aggressive. Keep low at small inference scales.",
		OBS_TEXT_INFO);

	p = obs_properties_add_int_slider(props, PROP_EDGE_BLUR,
		"Edge Blur", 0, 5, 1);
	obs_property_set_long_description(p,
		"Box blur on mask edges after erosion.");

	obs_properties_add_text(props, "info_blur",
		"Smooths jagged stair-step edges from the mask. "
		"1 = subtle  |  3 = soft",
		OBS_TEXT_INFO);

	// --- Temporal ---
	obs_properties_add_text(props, "info_temporal",
		"<b>Temporal</b>", OBS_TEXT_INFO);

	p = obs_properties_add_float_slider(props, PROP_SMOOTHING,
		"Temporal Smoothing", 0.0, 0.99, 0.01);
	obs_property_set_long_description(p,
		"Blends each frame's mask with the previous to reduce flickering.");

	obs_properties_add_text(props, "info_smoothing",
		"Reduces frame-to-frame jitter but adds lag. "
		"0 = off (fastest)  |  0.05 = subtle  |  0.5+ = smooth but laggy",
		OBS_TEXT_INFO);

	return props;
}

static void bg_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, PROP_DEVICE, 0);
	obs_data_set_default_double(settings, PROP_INFER_SCALE, 0.25);
	obs_data_set_default_double(settings, PROP_RESPONSIVENESS, 0.0);
	obs_data_set_default_double(settings, PROP_THRESHOLD, 0.15);
	obs_data_set_default_double(settings, PROP_FEATHER,   0.05);
	obs_data_set_default_int(settings, PROP_EROSION,   1);
	obs_data_set_default_int(settings, PROP_EDGE_BLUR, 1);
	obs_data_set_default_double(settings, PROP_SMOOTHING, 0.05);
}

static void bg_filter_update(void *data, obs_data_t *settings)
{
	bg_filter_data *f = static_cast<bg_filter_data *>(data);
	if (!f) return;
	std::lock_guard<std::mutex> lock(f->settings_mutex);
	f->smoothing = (float)obs_data_get_double(settings, PROP_SMOOTHING);
	f->threshold = (float)obs_data_get_double(settings, PROP_THRESHOLD);
	f->feather   = (float)obs_data_get_double(settings, PROP_FEATHER);
	f->erosion   = (int)obs_data_get_int(settings, PROP_EROSION);
	f->edge_blur = (int)obs_data_get_int(settings, PROP_EDGE_BLUR);
	f->infer_scale     = (float)obs_data_get_double(settings, PROP_INFER_SCALE);
	f->responsiveness  = (float)obs_data_get_double(settings, PROP_RESPONSIVENESS);
	f->device_id       = (int)obs_data_get_int(settings, PROP_DEVICE);
}

// ----------------------------------------------------------
// Create
// ----------------------------------------------------------
static void *bg_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "BG Removal: Creating filter...");
	bg_filter_data *f = nullptr;
	try { f = new bg_filter_data(); }
	catch (const std::bad_alloc &) {
		obs_log(LOG_ERROR, "BG Removal: Out of memory in create!");
		return nullptr;
	}
	f->source          = source;
	f->env             = nullptr;
	f->session         = nullptr;
	f->memory_info     = nullptr;
	f->texrender       = nullptr;
	f->small_texrender = nullptr;
	f->staging         = nullptr;
	f->mask_tex        = nullptr;
	f->mask_effect     = nullptr;
	f->linear_sampler  = nullptr;
	f->width           = 0;
	f->height          = 0;
	f->frame_linesize  = 0;
	f->frame_infer_w   = 0;
	f->frame_infer_h   = 0;
	f->mask_w          = 0;
	f->mask_h          = 0;
	f->initialized     = false;
	f->failed          = false;
	f->using_cuda      = false;
	f->frame_count     = 0;
	f->error_count     = 0;
	f->frame_ready.store(false);
	f->has_mask.store(false);
	f->running.store(false);
	f->infer_busy.store(false);
	f->smoothing       = 0.05f;
	f->threshold       = 0.15f;
	f->feather         = 0.05f;
	f->erosion         = 1;
	f->edge_blur       = 1;
	f->infer_scale     = 0.25f;
	f->responsiveness  = 0.0f;
	f->prev_infer_w    = 0;
	f->prev_infer_h    = 0;
	f->device_id       = 0;
	f->active_device_id = -2;  // sentinel: no live session yet

	bg_filter_update(f, settings);

	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	if (!f->texrender)
		obs_log(LOG_ERROR, "BG Removal: texrender failed!");
	else
		obs_log(LOG_INFO,  "BG Removal: texrender OK");

	f->small_texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	if (!f->small_texrender)
		obs_log(LOG_ERROR, "BG Removal: small_texrender failed!");

	// Linear sampler for bilinear mask upscale on GPU
	struct gs_sampler_info sinfo = {};
	sinfo.filter     = GS_FILTER_LINEAR;
	sinfo.address_u  = GS_ADDRESS_CLAMP;
	sinfo.address_v  = GS_ADDRESS_CLAMP;
	f->linear_sampler = gs_samplerstate_create(&sinfo);

	char *ep = obs_module_file("effects/mask.effect");
	if (!ep) {
		obs_log(LOG_ERROR, "BG Removal: mask.effect not found!");
	} else {
		obs_log(LOG_INFO, "BG Removal: Loading shader: %s", ep);
		f->mask_effect = gs_effect_create_from_file(ep, nullptr);
		bfree(ep);
		if (!f->mask_effect)
			obs_log(LOG_ERROR, "BG Removal: Shader failed!");
		else
			obs_log(LOG_INFO,  "BG Removal: Shader OK");
	}
	obs_leave_graphics();

	obs_log(LOG_INFO, "BG Removal: Filter created");
	return f;
}

// ----------------------------------------------------------
// Destroy
// ----------------------------------------------------------
static void bg_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "BG Removal: Destroying filter...");
	bg_filter_data *f = static_cast<bg_filter_data *>(data);
	if (!f) return;

	f->running.store(false);
	f->frame_cv.notify_all();
	if (f->infer_thread.joinable()) {
		obs_log(LOG_INFO, "BG Removal: Waiting for inference thread...");
		f->infer_thread.join();
	}

	obs_enter_graphics();
	if (f->texrender)       { gs_texrender_destroy(f->texrender);       f->texrender       = nullptr; }
	if (f->small_texrender) { gs_texrender_destroy(f->small_texrender); f->small_texrender = nullptr; }
	if (f->staging)         { gs_stagesurface_destroy(f->staging);      f->staging         = nullptr; }
	if (f->mask_tex)        { gs_texture_destroy(f->mask_tex);          f->mask_tex        = nullptr; }
	if (f->mask_effect)     { gs_effect_destroy(f->mask_effect);        f->mask_effect     = nullptr; }
	if (f->linear_sampler)  { gs_samplerstate_destroy(f->linear_sampler); f->linear_sampler = nullptr; }
	obs_leave_graphics();

	delete f->session;     f->session     = nullptr;
	delete f->memory_info; f->memory_info = nullptr;
	delete f->env;         f->env         = nullptr;
	delete f;

	obs_log(LOG_INFO, "BG Removal: Destroyed cleanly");
}

// ----------------------------------------------------------
// Render
// ----------------------------------------------------------
static void bg_filter_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	bg_filter_data *f = static_cast<bg_filter_data *>(data);
	if (!f || f->failed) {
		if (f) obs_source_skip_video_filter(f->source);
		return;
	}

	obs_source_t *target = obs_filter_get_target(f->source);
	if (!target) { obs_source_skip_video_filter(f->source); return; }

	uint32_t W = obs_source_get_base_width(target);
	uint32_t H = obs_source_get_base_height(target);
	// Sources can briefly report tiny/zero dimensions while warming up
	// (e.g. 1x1 before a webcam is ready). Treat anything below the minimum
	// as "not ready yet" and skip just this frame, so we retry next frame
	// instead of permanently failing the filter.
	if (W < MIN_WIDTH || H < MIN_HEIGHT || W > MAX_WIDTH || H > MAX_HEIGHT) {
		obs_source_skip_video_filter(f->source);
		return;
	}

	// Reinit on resolution change
	if (f->initialized && (W != f->width || H != f->height)) {
		obs_log(LOG_INFO, "BG Removal: Resolution changed %dx%d -> %dx%d",
			f->width, f->height, W, H);
		teardown_inference(f);
	}

	// Reinit on inference-device change (user picked a different GPU)
	int desired_dev;
	{
		std::lock_guard<std::mutex> lock(f->settings_mutex);
		desired_dev = f->device_id;
	}
	if (f->initialized && desired_dev != f->active_device_id) {
		obs_log(LOG_INFO,
			"BG Removal: Inference device changed %d -> %d, rebuilding session",
			f->active_device_id, desired_dev);
		teardown_inference(f);
	}

	if (!f->initialized) {
		if (!init_onnx(f, W, H)) {
			f->failed = true;
			obs_source_skip_video_filter(f->source);
			return;
		}
	}

	// Render source to offscreen (full res for display)
	if (!f->texrender) { obs_source_skip_video_filter(f->source); return; }
	gs_texrender_reset(f->texrender);
	if (!gs_texrender_begin(f->texrender, W, H)) {
		obs_source_skip_video_filter(f->source);
		return;
	}
	struct vec4 black; vec4_zero(&black);
	gs_clear(GS_CLEAR_COLOR, &black, 0.0f, 0);
	gs_ortho(0.0f, (float)W, 0.0f, (float)H, -100.0f, 100.0f);
	obs_source_video_render(target);
	gs_texrender_end(f->texrender);

	gs_texture_t *src_tex = gs_texrender_get_texture(f->texrender);
	if (!src_tex) { obs_source_skip_video_filter(f->source); return; }

	// Only stage when inference thread is idle (skip costly GPU->CPU copy otherwise)
	if (!f->infer_busy.load() && f->small_texrender) {
		// Compute inference dimensions
		float cur_scale;
		{
			std::lock_guard<std::mutex> lock(f->settings_mutex);
			cur_scale = f->infer_scale;
		}
		uint32_t IW = std::max((uint32_t)(W * cur_scale), (uint32_t)MIN_WIDTH);
		uint32_t IH = std::max((uint32_t)(H * cur_scale), (uint32_t)MIN_HEIGHT);
		IW &= ~1u;
		IH &= ~1u;

		// Render source to small texrender (GPU downscale - essentially free)
		gs_texrender_reset(f->small_texrender);
		if (gs_texrender_begin(f->small_texrender, IW, IH)) {
			gs_clear(GS_CLEAR_COLOR, &black, 0.0f, 0);
			gs_ortho(0.0f, (float)W, 0.0f, (float)H, -100.0f, 100.0f);
			obs_source_video_render(target);
			gs_texrender_end(f->small_texrender);

			gs_texture_t *small_tex = gs_texrender_get_texture(f->small_texrender);
			if (small_tex) {
				// Recreate staging surface if size changed
				if (!f->staging ||
				    gs_stagesurface_get_width(f->staging)  != IW ||
				    gs_stagesurface_get_height(f->staging) != IH) {
					if (f->staging) {
						gs_stagesurface_destroy(f->staging);
						f->staging = nullptr;
					}
					f->staging = gs_stagesurface_create(IW, IH, GS_BGRA);
				}

				if (f->staging) {
					gs_stage_texture(f->staging, small_tex);
					uint8_t *mapped = nullptr;
					uint32_t linesize = 0;
					if (gs_stagesurface_map(f->staging, &mapped, &linesize)
					    && mapped && linesize) {
						std::lock_guard<std::mutex> lock(f->frame_mutex);
						size_t total = (size_t)IH * linesize;
						f->frame_bgra.resize(total);
						memcpy(f->frame_bgra.data(), mapped, total);
						f->frame_linesize = linesize;
						f->frame_infer_w  = IW;
						f->frame_infer_h  = IH;
						f->frame_ready.store(true);
						f->frame_cv.notify_one();
					}
					gs_stagesurface_unmap(f->staging);
				}
			}
		}
	}

	// Render mask if available
	if (f->has_mask.load()) {
		std::vector<uint8_t> local_mask;
		uint32_t mw, mh;
		{
			std::lock_guard<std::mutex> lock(f->mask_mutex);
			local_mask = f->mask_rgba;
			mw = f->mask_w;
			mh = f->mask_h;
		}

		if (!local_mask.empty() && mw && mh) {
			// Create mask texture at inference resolution (small)
			if (!f->mask_tex ||
			    gs_texture_get_width(f->mask_tex)  != mw ||
			    gs_texture_get_height(f->mask_tex) != mh) {
				if (f->mask_tex) {
					gs_texture_destroy(f->mask_tex);
					f->mask_tex = nullptr;
				}
				f->mask_tex = gs_texture_create(
					mw, mh, GS_BGRA, 1, nullptr, GS_DYNAMIC);
				if (!f->mask_tex) {
					obs_log(LOG_ERROR, "BG Removal: mask_tex failed!");
					obs_source_skip_video_filter(f->source);
					return;
				}
			}

			gs_texture_set_image(f->mask_tex,
					     local_mask.data(), mw * 4, false);

			gs_eparam_t *p_img = gs_effect_get_param_by_name(
				f->mask_effect, "image");
			gs_eparam_t *p_msk = gs_effect_get_param_by_name(
				f->mask_effect, "mask_tex");

			if (!p_img || !p_msk) {
				obs_source_skip_video_filter(f->source);
				return;
			}

			gs_blend_state_push();
			gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
			gs_effect_set_texture(p_img, src_tex);
			// Set linear sampler so GPU does bilinear upscale
			gs_effect_set_next_sampler(p_msk, f->linear_sampler);
			gs_effect_set_texture(p_msk, f->mask_tex);
			while (gs_effect_loop(f->mask_effect, "Draw"))
				gs_draw_sprite(src_tex, 0, W, H);
			gs_blend_state_pop();
			return;
		}
	}

	obs_source_skip_video_filter(f->source);
}

// ----------------------------------------------------------
// Register with OBS
// ----------------------------------------------------------
extern "C" struct obs_source_info bg_removal_filter = {
	.id             = "bg_removal_filter",
	.type           = OBS_SOURCE_TYPE_FILTER,
	.output_flags   = OBS_SOURCE_VIDEO,
	.get_name       = bg_filter_name,
	.create         = bg_filter_create,
	.destroy        = bg_filter_destroy,
	.get_defaults   = bg_filter_defaults,
	.get_properties = bg_filter_properties,
	.update         = bg_filter_update,
	.video_render   = bg_filter_render,
};
