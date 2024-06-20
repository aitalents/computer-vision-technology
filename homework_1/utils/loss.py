import torch

def yolo_loss(yhat, y, lambda_coord=5, lambda_noobj=0.5, n_classes=80):
	"""
	Args:
		yhat: [#, 7, 7, 30]
		y: [#, 7, 7, 30]
	Returns:
		loss: [#]
	"""
	with torch.no_grad():
		# arrange cell xidx, yidx
		# [7, 7]
		cell_xidx = (torch.arange(49) % 7).reshape(7, 7)
		cell_yidx = (torch.div(torch.arange(49), 7, rounding_mode='floor')).reshape(7, 7)
		# transform to [7, 7, 2]
		cell_xidx.unsqueeze_(-1)
		cell_yidx.unsqueeze_(-1)
		cell_xidx.expand(7, 7, 2)
		cell_yidx.expand(7, 7, 2)
		# move to device
		cell_xidx = cell_xidx.to(yhat.device)
		cell_yidx = cell_yidx.to(yhat.device)

	def calc_coord(val):
		with torch.no_grad():
			# transform cell relative coordinates to image relative coordinates
			x = (val[..., 0] + cell_xidx) / 7.0
			y = (val[..., 1] + cell_yidx) / 7.0

			return (x - val[..., 2] / 2.0,
				x + val[..., 2] / 2.0,
				y - val[..., 3] / 2.0,
				y + val[..., 3] / 2.0)

	y_area = y[..., :10].reshape(-1, 7, 7, 2, 5)
	yhat_area = yhat[..., :10].reshape(-1, 7, 7, 2, 5)

	y_class = y[..., 10:].reshape(-1, 7, 7, n_classes)
	yhat_class = yhat[..., 10:].reshape(-1, 7, 7, n_classes)

	with torch.no_grad():
		# calculate IoU
		x_min, x_max, y_min, y_max = calc_coord(y_area)
		x_min_hat, x_max_hat, y_min_hat, y_max_hat = calc_coord(yhat_area)

		wi = torch.min(x_max, x_max_hat) - torch.max(x_min, x_min_hat)
		wi = torch.max(wi, torch.zeros_like(wi))
		hi = torch.min(y_max, y_max_hat) - torch.max(y_min, y_min_hat)
		hi = torch.max(hi, torch.zeros_like(hi))

		intersection = wi * hi
		union = (x_max - x_min) * (y_max - y_min) + (x_max_hat - x_min_hat) * (y_max_hat - y_min_hat) - intersection
		iou = intersection / (union + 1e-6) # add epsilon to avoid nan

		_, res = iou.max(dim=3, keepdim=True)

	# [#, 7, 7, 5]
	# responsible bounding box (having higher IoU)
	yhat_res = torch.take_along_dim(yhat_area, res.unsqueeze(3), 3).squeeze_(3)
	y_res = y_area[..., 0, :5]

	with torch.no_grad():
		# calculate indicator matrix
		have_obj = y_res[..., 4] > 0
		no_obj = ~have_obj
		
	return ((lambda_coord * ( # coordinate loss
		  (y_res[..., 0] - yhat_res[..., 0]) ** 2 # X
		+ (y_res[..., 1] - yhat_res[..., 1]) ** 2 # Y
		+ (torch.sqrt(y_res[..., 2]) - torch.sqrt(yhat_res[..., 2])) ** 2  # W
		+ (torch.sqrt(y_res[..., 3]) - torch.sqrt(yhat_res[..., 3])) ** 2) # H
		# confidence
		+ (y_res[..., 4] - yhat_res[..., 4]) ** 2
		# class
		+ ((y_class - yhat_class) ** 2).sum(dim=3)) * have_obj
		# noobj
		+ ((y_area[..., 0, 4] - yhat_area[..., 0, 4]) ** 2 + \
		(y_area[..., 1, 4] - yhat_area[..., 1, 4]) ** 2) * no_obj * lambda_noobj).sum(dim=(1, 2))
