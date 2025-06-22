#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json
import numpy as np
import shutil
import time
import datetime  
import cv2  

from common import *
from scenes import *

from tqdm import tqdm
import pyngp as ngp  # noqa
from tensorboardX import SummaryWriter  # For TensorBoard logging

MAX_EVAL_RESOLUTION = 8192  


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run instant-ngp: train, evaluate, and output assets with TensorBoard logging.")

	# Arguments from original run.py
	parser.add_argument("files", nargs="*",
						help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")
	# ... (all original arguments from your provided code) ...
	parser.add_argument("--scene", "--training_data", default="",
						help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS)  # deprecated
	parser.add_argument("--network", default="",
						help="Path to the network config. Uses the scene's default if unspecified.")
	parser.add_argument("--load_snapshot", "--snapshot", default="",
						help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--save_snapshot", default="",
						help="Save this snapshot after training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF.")
	parser.add_argument("--test_transforms", default="",
						help="Path to a NeRF-style transforms.json for test set evaluation.")
	parser.add_argument("--near_distance", default=-1, type=float,
						help="Set the distance from the camera at which training rays start for NeRF. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float,
						help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")
	parser.add_argument("--screenshot_transforms", default="",
						help="Path to a NeRF-style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")
	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true",
						help="Applies additional smoothing to the camera trajectory.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1,
						help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1),
						metavar=("START_FRAME", "END_FRAME"),
						help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	parser.add_argument("--video_spp", type=int, default=8,
						help="Number of samples per pixel in video. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4",
						help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")
	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh. Supports OBJ and PLY.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Resolution for the marching cubes grid.")
	parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float,
						help="Density threshold for marching cubes.")
	parser.add_argument("--width", "--screenshot_w", type=int, default=0,
						help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0,
						help="Resolution height of GUI and screenshots.")
	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true",
						help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true",
						help="Open a second window with a copy of the main output.")
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")
	parser.add_argument("--sharpen", default=0, help="Amount of sharpening for NeRF training images (0.0 to 1.0).")
	parser.add_argument("--eval_interval", type=int, default=0,
						help="Frequency to save a checkpoint, in steps. 0=disabled.")

	parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs.")

	parser.add_argument("--eval_spp", type=int, default=4,
						help="Samples per pixel for evaluation rendering. Lower for faster, less accurate eval.")
	parser.add_argument("--eval_max_res_override", type=int, default=MAX_EVAL_RESOLUTION,
						help="Override max dimension for evaluation images. Lower for faster eval. Default: %(default)s")
	parser.add_argument("--eval_num_images", type=int, default=-1,
						help="Number of images to evaluate from test_transforms.json. -1 for all. Fewer for faster eval.")
	parser.add_argument("--skip_eval_tb_images", action="store_true",
						help="Skip saving evaluation comparison images to TensorBoard to save a little time.")
	parser.add_argument("--eval_pixel_fraction", type=float, default=1.0,
						help="Fraction of pixels to use for MSE/PSNR calculation during eval (0.0 to 1.0]. "
							 "Applied after center crop if any. SSIM calculated on (cropped) full image. Default: %(default)s")
	parser.add_argument("--eval_center_crop_ratio", type=float, default=1.0,
						help="Ratio of image to use from center for evaluation (0.0 to 1.0]. "
							 "1.0 means use full image. Applied before pixel fraction. Default: %(default)s")

	return parser.parse_args()


def get_scene(scene_name_or_path):
	for scenes_dict in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene_name_or_path in scenes_dict:
			return scenes_dict[scene_name_or_path]
	if os.path.exists(scene_name_or_path):
		return {"data_dir": os.path.dirname(scene_name_or_path), "dataset": os.path.basename(scene_name_or_path)}
	return None


def perform_evaluation(testbed, test_transforms_path, writer, eval_step,
					   max_resolution_dim_override, eval_spp, eval_num_images, skip_tb_images,
					   eval_pixel_fraction, eval_center_crop_ratio,  # NEW ARGS
					   scene_name_for_log="eval"):
	print(f"\n--- Evaluating model at step {eval_step} on test data: {test_transforms_path} ---")
	eval_notes = [f"SPP={eval_spp}", f"MaxRes={max_resolution_dim_override}"]
	if eval_num_images > 0:
		eval_notes.append(f"NumImgs={eval_num_images}")
	if 1.0 > eval_center_crop_ratio > 0.0:
		eval_notes.append(f"CenterCrop={eval_center_crop_ratio * 100:.0f}%")
	if 1.0 > eval_pixel_fraction > 0.0:
		eval_notes.append(f"PixFrac={eval_pixel_fraction * 100:.0f}% MSE/PSNR")
	print(f"    Eval settings: {', '.join(eval_notes)}")

	try:
		with open(test_transforms_path, 'r') as f:
			test_transforms = json.load(f)
	except Exception as e:
		print(f"Error loading test_transforms.json: {e}")
		return

	original_background_color = testbed.background_color
	original_snap_to_pixel_centers = testbed.snap_to_pixel_centers
	original_render_min_transmittance = testbed.nerf.render_min_transmittance

	testbed.background_color = [0.0, 0.0, 0.0, 1.0]
	testbed.snap_to_pixel_centers = True
	testbed.nerf.render_min_transmittance = 1e-3

	tot_mse, tot_psnr, tot_ssim = 0, 0, 0
	count_mse_psnr, count_ssim = 0, 0  # Separate counters for robust averaging

	test_frames_all = test_transforms.get("frames", [])
	if not test_frames_all:
		print("Warning: No 'frames' found in test_transforms.json.")
		# Restore settings before returning
		testbed.background_color = original_background_color
		testbed.snap_to_pixel_centers = original_snap_to_pixel_centers
		testbed.nerf.render_min_transmittance = original_render_min_transmittance
		testbed.shall_train = True
		return

	if 0 < eval_num_images < len(test_frames_all):
		test_frames = test_frames_all[:eval_num_images]
		print(f"    Evaluating on the first {len(test_frames)} images instead of all {len(test_frames_all)}.")
	else:
		test_frames = test_frames_all

	data_dir = os.path.dirname(test_transforms_path)
	testbed.shall_train = False

	for i, frame_data in enumerate(tqdm(test_frames, unit="images", desc="  Rendering test frames")):
		img_path = os.path.join(data_dir, frame_data["file_path"])
		if not os.path.splitext(img_path)[1]:
			img_path += ".png"

		try:
			ref_image_bgr = cv2.imread(img_path)
			if ref_image_bgr is None:
				raise IOError(f"Could not read image: {img_path}")

			original_h, original_w = ref_image_bgr.shape[:2]

			if max(original_w, original_h) > max_resolution_dim_override:
				scale_factor = max_resolution_dim_override / max(original_w, original_h)
				new_w = int(original_w * scale_factor)
				new_h = int(original_h * scale_factor)
				ref_image_bgr_resized = cv2.resize(ref_image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
				render_resolution = [new_w, new_h]
			else:
				ref_image_bgr_resized = ref_image_bgr
				render_resolution = [original_w, original_h]

			# Convert GT to linear RGB float [0,1] for consistent processing with prediction
			ref_image_rgb_linear_gt = cv2.cvtColor(ref_image_bgr_resized, cv2.COLOR_BGR2RGB) / 255.0

		except Exception as e:
			print(f"Warning: Could not load or resize ground truth image {img_path}: {e}")
			continue

		cam_matrix = np.matrix(frame_data["transform_matrix"])
		testbed.set_nerf_camera_matrix(cam_matrix[:-1, :])

		if "camera_angle_x" in test_transforms:
			testbed.fov_axis = 0
			testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		elif "camera_angle_y" in test_transforms:
			testbed.fov_axis = 1
			testbed.fov = test_transforms["camera_angle_y"] * 180 / np.pi

		image_rgba_linear = testbed.render(render_resolution[0], render_resolution[1], eval_spp, True)

		pred_image_srgb = np.clip(linear_to_srgb(image_rgba_linear[..., :3]), 0.0, 1.0)
		ref_image_srgb_gt = np.clip(linear_to_srgb(ref_image_rgb_linear_gt), 0.0, 1.0)

		current_pred_srgb = pred_image_srgb
		current_ref_srgb_gt = ref_image_srgb_gt

		if 0.0 < eval_center_crop_ratio < 1.0:
			h, w = current_pred_srgb.shape[:2]
			crop_h, crop_w = int(h * eval_center_crop_ratio), int(w * eval_center_crop_ratio)
			start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
			current_pred_srgb = current_pred_srgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
			current_ref_srgb_gt = current_ref_srgb_gt[start_h:start_h + crop_h, start_w:start_w + crop_w]
			if i == 0:
				print(f"Applied center crop. New eval HxW: {crop_h}x{crop_w}")

		ssim_pred_img = current_pred_srgb
		ssim_ref_img = current_ref_srgb_gt

		mse_psnr_pred_pixels = current_pred_srgb
		mse_psnr_ref_pixels = current_ref_srgb_gt
		is_pixel_sampled_for_mse_psnr = False

		if 0.0 < eval_pixel_fraction < 1.0:
			is_pixel_sampled_for_mse_psnr = True
			h, w = mse_psnr_pred_pixels.shape[:2]
			if h * w == 0:  
				print(f"Warning: Image {i} has zero dimension after cropping. Skipping metrics for this image.")
				continue

			num_pixels_to_sample = int(h * w * eval_pixel_fraction)
			if num_pixels_to_sample == 0 and h * w > 0:  # Ensure at least 1 pixel if original image not empty
				num_pixels_to_sample = 1

			if num_pixels_to_sample > 0:
				y_indices = np.random.randint(0, h, num_pixels_to_sample)
				x_indices = np.random.randint(0, w, num_pixels_to_sample)
				mse_psnr_pred_pixels = mse_psnr_pred_pixels[y_indices, x_indices]
				mse_psnr_ref_pixels = mse_psnr_ref_pixels[y_indices, x_indices]
				if i == 0:
					print(
						f"Using {num_pixels_to_sample} ({eval_pixel_fraction * 100:.1f}%) pixels for MSE/PSNR."
					)
			else:
				print(f"Warning: No pixels to sample for MSE/PSNR for image {i}. Skipping MSE/PSNR.")
				mse_psnr_pred_pixels = None

		mse = -1.0
		psnr = -1.0
		if mse_psnr_pred_pixels is not None and mse_psnr_ref_pixels is not None and mse_psnr_pred_pixels.size > 0:
			try:
				mse = np.mean((mse_psnr_pred_pixels - mse_psnr_ref_pixels) ** 2)
				psnr = mse2psnr(mse)
				tot_mse += mse
				tot_psnr += psnr
				count_mse_psnr += 1
			except Exception as e_metric:
				print(f"Warning: Could not compute MSE/PSNR for image {i}: {e_metric}")

		ssim = -1.0
		try:
			ssim = float(compute_error("SSIM", ssim_pred_img, ssim_ref_img))
			if is_pixel_sampled_for_mse_psnr and i == 0 and ssim >= 0:
				print(f"    (Note: SSIM calculated on denser region than MSE/PSNR)")
			tot_ssim += ssim
			count_ssim += 1
		except Exception as e_ssim:
			print(f"Warning: SSIM calculation failed for image {i}: {e_ssim}")

		if i == 0 and writer and not skip_tb_images:
			try:
				img_to_log_pred = (pred_image_srgb * 255).astype(np.uint8)
				img_to_log_gt = (ref_image_srgb_gt * 255).astype(np.uint8)
				diff_img = np.abs(pred_image_srgb - ref_image_srgb_gt)
				diff_img_to_log = (diff_img / diff_img.max() * 255).astype(np.uint8) if diff_img.max() > 0 else (
					diff_img * 255).astype(np.uint8)

				writer.add_image(
					f'Eval/{scene_name_for_log}/Frame_{i}/Predicted_sRGB', img_to_log_pred, eval_step, dataformats='HWC'
				)
				writer.add_image(
					f'Eval/{scene_name_for_log}/Frame_{i}/GroundTruth_sRGB', img_to_log_gt, eval_step, dataformats='HWC'
				)
				writer.add_image(
					f'Eval/{scene_name_for_log}/Frame_{i}/Difference_sRGB', diff_img_to_log, eval_step, dataformats='HWC'
				)
			except Exception as e_img:
				print(f"Warning: could not log images to TensorBoard: {e_img}")

	avg_mse_str, avg_psnr_str, avg_ssim_str = "N/A", "N/A", "N/A"
	if count_mse_psnr > 0:
		avg_mse = tot_mse / count_mse_psnr
		avg_psnr = tot_psnr / count_mse_psnr
		avg_mse_str = f"{avg_mse:.6f}"
		avg_psnr_str = f"{avg_psnr:.2f}"
		if writer:
			writer.add_scalar(f'Loss/test_mse_{scene_name_for_log}', avg_mse, eval_step)
			writer.add_scalar(f'Metrics/test_psnr_{scene_name_for_log}', avg_psnr, eval_step)
	else:
		print("  No images were successfully evaluated for MSE/PSNR.")

	if count_ssim > 0:
		avg_ssim = tot_ssim / count_ssim
		avg_ssim_str = f"{avg_ssim:.4f}"
		if writer:
			writer.add_scalar(f'Metrics/test_ssim_{scene_name_for_log}', avg_ssim, eval_step)
	else:
		print("  No images were successfully evaluated for SSIM.")

	print(f"  Results (avg over up to {len(test_frames)} attempted images):")
	print(
		f"  MSE: {avg_mse_str} (from {count_mse_psnr} imgs), PSNR: {avg_psnr_str} (from {count_mse_psnr} imgs), "
		f"SSIM: {avg_ssim_str} (from {count_ssim} imgs)"
	)

	testbed.background_color = original_background_color
	testbed.snap_to_pixel_centers = original_snap_to_pixel_centers
	testbed.nerf.render_min_transmittance = original_render_min_transmittance
	testbed.shall_train = True


if __name__ == "__main__":
	args = parse_args()

	if args.vr:
		args.gui = True
	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect.")

	scene_name_for_logs = "unnamed_scene"
	if args.scene:
		scene_name_for_logs = os.path.splitext(os.path.basename(args.scene))[0]
	elif args.load_snapshot:
		scene_name_for_logs = os.path.splitext(os.path.basename(args.load_snapshot))[0]

	timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	tb_log_dir = os.path.join(args.log_dir, f"{scene_name_for_logs}_{timestamp}")
	writer = SummaryWriter(tb_log_dir)
	print(f"TensorBoard logs will be saved to: {tb_log_dir}")

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	for file_path in args.files:
		scene_info = get_scene(file_path)
		if scene_info:
			resolved_path = os.path.join(scene_info["data_dir"], scene_info["dataset"])
		else:
			resolved_path = file_path
		testbed.load_file(resolved_path)

	if args.scene:
		scene_info = get_scene(args.scene)
		if scene_info is not None:
			args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
			if not args.network and "network" in scene_info:
				args.network = scene_info["network"]
		testbed.load_training_data(args.scene)

	if args.gui:
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window)
		if args.vr:
			testbed.init_vr()

	if args.load_snapshot:
		scene_info = get_scene(args.load_snapshot)
		if scene_info is not None:
			args.load_snapshot = default_snapshot_filename(scene_info)
		testbed.load_snapshot(args.load_snapshot)

	elif args.network:
		testbed.reload_network_from_file(args.network)

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True
	testbed.nerf.render_with_lens_distortion = True

	network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
	if testbed.mode == ngp.TestbedMode.Sdf:
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print(f"Setting NeRF training ray near_distance: {args.near_distance}")
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print("NeRF compatibility mode enabled.")
		testbed.color_space = ngp.ColorSpace.SRGB
		testbed.nerf.cone_angle_constant = 0
		testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0:
		n_steps = 35000 if not args.load_snapshot or args.gui else 0

	tqdm_last_update_time = time.monotonic()
	last_periodic_save_step = -1
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="steps") as t:
			while testbed.frame():
				current_step = testbed.training_step

				if testbed.want_repl():
					repl(testbed)

				if current_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				now = time.monotonic()
				if now - tqdm_last_update_time > 0.1:
					t.update(current_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					writer.add_scalar(f'Loss/train_{scene_name_for_logs}', testbed.loss, current_step)
					old_training_step = current_step
					tqdm_last_update_time = now

				if args.eval_interval > 0 and args.test_transforms and current_step > 0 and \
					(current_step % args.eval_interval == 0):
					if last_periodic_save_step != current_step:
						perform_evaluation(
							testbed,
							args.test_transforms,
							writer,
							testbed.training_step,
							args.eval_max_res_override,
							args.eval_spp,
							args.eval_num_images,
							args.skip_eval_tb_images,
							args.eval_pixel_fraction,
							args.eval_center_crop_ratio,
							scene_name_for_logs
						)
						last_periodic_save_step = current_step

	final_snapshot_path_abs = None
	if args.save_snapshot:
		# Create directory if it does not exist
		snapshot_dir = os.path.dirname(args.save_snapshot)
		if snapshot_dir and not os.path.exists(snapshot_dir):
			os.makedirs(snapshot_dir, exist_ok=True)
		testbed.save_snapshot(args.save_snapshot, False)  # save_snapshot might fail if dir doesn't exist
		final_snapshot_path_abs = os.path.abspath(args.save_snapshot)
		print(f"Final snapshot saved to: {args.save_snapshot}")

	if args.test_transforms and (
		args.eval_interval == 0 or n_steps == 0 or last_periodic_save_step != testbed.training_step):
		print("\n--- Performing Final Evaluation ---")
		perform_evaluation(
			testbed,
			args.test_transforms,
			writer,
			testbed.training_step,  # Use current (final) step
			args.eval_max_res_override,
			args.eval_spp,
			args.eval_num_images,  # Or you might want to use full eval here, e.g., -1 for eval_num_images
			args.skip_eval_tb_images,
			args.eval_pixel_fraction,  # Pass new arg
			args.eval_center_crop_ratio,  # Pass new arg
			scene_name_for_logs
		)

	if args.save_mesh:
		res = args.marching_cubes_res
		thresh = args.marching_cubes_density_thresh
		print(
			f"Generating mesh via marching cubes: {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)

	screenshot_ref_transforms = {}
	if args.screenshot_transforms:
		print(f"Loading screenshot transforms from: {args.screenshot_transforms}")
		with open(args.screenshot_transforms) as f:
			screenshot_ref_transforms = json.load(f)

	if screenshot_ref_transforms:
		testbed.fov_axis = 0
		if "camera_angle_x" in screenshot_ref_transforms:
			testbed.fov = screenshot_ref_transforms["camera_angle_x"] * 180 / np.pi
		elif "camera_angle_y" in screenshot_ref_transforms:
			testbed.fov_axis = 1
			testbed.fov = screenshot_ref_transforms["camera_angle_y"] * 180 / np.pi

		frames_to_render = args.screenshot_frames
		if not frames_to_render:
			frames_to_render = range(len(screenshot_ref_transforms["frames"]))

		print(f"Rendering screenshot frames: {list(map(str, frames_to_render))}")
		for idx_str in frames_to_render:
			idx = int(idx_str)
			if idx >= len(screenshot_ref_transforms["frames"]):
				print(f"Warning: screenshot frame index {idx} out of bounds.")
				continue

			frame_data = screenshot_ref_transforms["frames"][idx]
			cam_matrix = np.matrix(frame_data.get("transform_matrix", frame_data.get(
				"transform_matrix_start")))
			testbed.set_nerf_camera_matrix(cam_matrix[:-1, :])

			outname = os.path.join(args.screenshot_dir or ".", os.path.basename(frame_data["file_path"]))
			if not os.path.splitext(outname)[1]:
				outname += ".png"

			print(f"Rendering screenshot: {outname}")
			render_w = args.width or int(screenshot_ref_transforms.get("w", testbed.width))
			render_h = args.height or int(screenshot_ref_transforms.get("h", testbed.height))

			image = testbed.render(render_w, render_h, args.screenshot_spp, True)
			if os.path.dirname(outname):
				os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)

	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, f"{scene_name_for_logs}_{network_stem}_screenshot.png")
		print(f"Rendering single screenshot: {outname}")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname):
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname, image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)
		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps
		save_individual_frames = "%" in args.video_output
		start_frame, end_frame = args.video_render_range

		temp_video_dir = "tmp_video_frames"
		if not save_individual_frames:
			if os.path.exists(temp_video_dir):
				shutil.rmtree(temp_video_dir)
			os.makedirs(temp_video_dir, exist_ok=True)

		print(f"Rendering video to: {args.video_output}")
		for i in tqdm(range(n_frames), unit="frames", desc="Rendering video frames"):
			testbed.camera_smoothing = args.video_camera_smoothing
			if start_frame >= 0 and i < start_frame:
				frame = testbed.render(
					32, 32, 1, True, float(i) / n_frames, float(i + 1) / n_frames, args.video_fps, shutter_fraction=0.5
				)
				continue
			if i > end_frame >= 0:
				continue

			frame = testbed.render(
				resolution[0], resolution[1], args.video_spp, True, float(i) / n_frames,
				float(i + 1) / n_frames, args.video_fps, shutter_fraction=0.5
			)
			processed_frame = np.clip(frame * (2 ** args.exposure), 0.0, 1.0)

			if save_individual_frames:
				frame_output_path = args.video_output % i
				if os.path.dirname(frame_output_path):
					os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
				write_image(frame_output_path, processed_frame, quality=100)
			else:
				write_image(os.path.join(temp_video_dir, f"{i:04d}.jpg"), processed_frame, quality=100)

		if not save_individual_frames:
			ffmpeg_cmd = (
				f"ffmpeg -y -framerate {args.video_fps} -i {temp_video_dir}/%04d.jpg "
				f"-c:v libx264 -pix_fmt yuv420p {args.video_output}"
			)
			print(f"Running ffmpeg: {ffmpeg_cmd}")
			os.system(ffmpeg_cmd)
			shutil.rmtree(temp_video_dir)
		print("Video rendering complete.")

	writer.close()
	print("--- Script finished ---")
