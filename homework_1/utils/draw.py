import matplotlib.pyplot as plt
from typing import Optional


def draw_precision_recall(pr_data: list, class_idx: Optional[int]=None):
	p = [data['precision'] for data in pr_data]
	r = [data['recall'] for data in pr_data]

	plt.plot(r, p, 'o-', color='r')
	plt.xlabel("Recall")
	plt.ylabel("Precision")

	if class_idx is not None:
		plt.title(categories[class_idx])

	plt.show()
