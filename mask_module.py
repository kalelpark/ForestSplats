def generate_filled_mask(image: np.ndarray, mask: np.ndarray, num_segments=20) -> np.ndarray:
    # SLIC Superpixel Segmentation
    segments = slic(image, n_segments=num_segments, compactness=1, sigma=3, start_label=0)
    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for region in regionprops(segments):
        coords = region.coords  
        
        coords[:, 0] = np.clip(coords[:, 0], 0, mask.shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, mask.shape[1] - 1)
        mask_pixels = mask[coords[:, 0], coords[:, 1]]

        if isinstance(mask_pixels, torch.Tensor):
            mask_pixels = mask_pixels.cpu().numpy()
        mask_ratio = np.sum(mask_pixels) / len(mask_pixels)
        
        if mask_ratio >= 0.2:
            region_mask = np.zeros_like(mask, dtype=np.float32)
            region_mask[coords[:, 0], coords[:, 1]] = mask[coords[:, 0], coords[:, 1]]

            region_mask_torch = torch.tensor(region_mask).unsqueeze(0).unsqueeze(0)
            h, w = mask.shape
            grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            grid = np.stack((grid_x, grid_y), axis=-1)
            grid_torch = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1, h, w, 2)

            interpolated_mask = F.grid_sample(region_mask_torch, grid_torch, mode='bilinear', align_corners=False)
            interpolated_mask_np = interpolated_mask.squeeze().numpy()

            filled_mask[coords[:, 0], coords[:, 1]] = np.max(interpolated_mask_np[coords[:, 0], coords[:, 1]]).astype(np.uint8)
        else:
            filled_mask[coords[:, 0], coords[:, 1]] = mask[coords[:, 0], coords[:, 1]]
    
    return filled_mask
