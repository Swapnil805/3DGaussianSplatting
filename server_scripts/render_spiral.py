import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from tqdm import tqdm
from imageio import imwrite
from argparse import ArgumentParser
import copy

from scene import GaussianModel, Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def strip_camera_images(cams):
    for c in cams:
        for attr in [
            "original_image",
            "image",
            "gt_alpha_mask",
            "alpha_mask",
            "invdepthmap",
            "depth",
            "mask",
        ]:
            if hasattr(c, attr):
                try:
                    setattr(c, attr, None)
                except Exception:
                    pass


def build_c2w_from_lookat(cam_pos, target, up=np.array([0, 1, 0], dtype=np.float32), ref_R=None):
    cam_pos = cam_pos.astype(np.float32)
    target = target.astype(np.float32)

    def make_R(forward_world):
        forward_world = forward_world / (np.linalg.norm(forward_world) + 1e-8)
        right_world = np.cross(forward_world, up)
        right_world = right_world / (np.linalg.norm(right_world) + 1e-8)
        up_world = np.cross(right_world, forward_world)
        up_world = up_world / (np.linalg.norm(up_world) + 1e-8)
        return np.stack([right_world, up_world, forward_world], axis=1).astype(np.float32)

    f1 = target - cam_pos
    f2 = cam_pos - target

    R1 = make_R(f1)
    R2 = make_R(f2)

    if ref_R is None:
        return R1

    d1 = np.linalg.norm(R1 - ref_R)
    d2 = np.linalg.norm(R2 - ref_R)
    return R1 if d1 <= d2 else R2


def main():
    safe_state(False)
    device = torch.device("cuda")

    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    args.model_path = "/workspace/gaussian_project/output_model"
    args.source_path = "/workspace/gaussian_project/scene_gs"
    trained_iter = 40000
    out_dir = "/workspace/gaussian_project/results/novel_spiral"

    os.makedirs(out_dir, exist_ok=True)

    args.data_device = "cpu"
    args.resolution = 2

    tmp_gaussians = GaussianModel(lp.sh_degree)
    scene = Scene(lp.extract(args), tmp_gaussians)

    train_cams = scene.getTrainCameras()
    strip_camera_images(train_cams)

    base_cam = train_cams[0]
    base_cam.world_view_transform = base_cam.world_view_transform.cuda()
    base_cam.projection_matrix = base_cam.projection_matrix.cuda()
    base_cam.full_proj_transform = base_cam.full_proj_transform.cuda()
    base_cam.camera_center = base_cam.camera_center.cuda()

    centers = np.stack(
        [c.camera_center.detach().cpu().numpy() for c in train_cams],
        axis=0,
    ).astype(np.float32)

    scene_center = centers.mean(axis=0)
    mean_dist = np.linalg.norm(centers - scene_center[None, :], axis=1).mean()
    base_radius = float(mean_dist * 2.0)

    del tmp_gaussians
    torch.cuda.empty_cache()

    gaussians = GaussianModel(lp.sh_degree)

    ply_path = os.path.join(
        args.model_path,
        "point_cloud",
        f"iteration_{trained_iter}",
        "point_cloud.ply",
    )

    gaussians.load_ply(ply_path)

    pipe = pp.extract(args)
    pipe.resolution_scale = 1.0

    bg = torch.zeros(3, device=device)
    num_frames = 240

    print(f"Scene center: {scene_center.tolist()}")
    print(f"Base radius: {base_radius:.3f}")
    print("Rendering SPIRAL views...")

    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            t = i / max(num_frames - 1, 1)
            theta = 2.0 * np.pi * t
            radius_t = base_radius * (0.7 + 0.6 * t)

            height_amp = base_radius * 0.3
            y = scene_center[1] + height_amp * (t - 0.5) * 2.0

            cam_pos = np.array(
                [
                    scene_center[0] + radius_t * np.cos(theta),
                    y,
                    scene_center[2] + radius_t * np.sin(theta),
                ],
                dtype=np.float32,
            )

            ref_R = base_cam.R.astype(np.float32) if hasattr(base_cam, "R") else None
            R_c2w = build_c2w_from_lookat(cam_pos, scene_center, ref_R=ref_R)
            T = (-R_c2w.T @ cam_pos).astype(np.float32)

            cam = copy.deepcopy(base_cam)
            cam.R = R_c2w
            cam.T = T
            cam.image_name = f"spiral_{i:04d}.png"

            wv = getWorld2View2(cam.R, cam.T)
            cam.world_view_transform = torch.tensor(wv, device=device).transpose(0, 1)

            proj = getProjectionMatrix(
                znear=cam.znear,
                zfar=cam.zfar,
                fovX=cam.FoVx,
                fovY=cam.FoVy,
            )

            cam.projection_matrix = torch.tensor(proj, device=device).transpose(0, 1)
            cam.full_proj_transform = cam.world_view_transform @ cam.projection_matrix
            cam.camera_center = torch.tensor(cam_pos, device=device)

            pkg = render(cam, gaussians, pipe, bg)
            image = pkg["render"]

            img = (
                torch.clamp(image, 0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            imwrite(os.path.join(out_dir, cam.image_name), img)

            del pkg, image, img, cam

            if i % 30 == 0:
                torch.cuda.empty_cache()

    print("Spiral rendering complete!")


if __name__ == "__main__":
    main()
