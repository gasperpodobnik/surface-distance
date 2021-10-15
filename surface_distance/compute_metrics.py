from . import metrics
import SimpleITK as sitk
import numpy as np


class compute_metrics_deepmind:
    def __init__(self, organs_labels_dict: dict, metrics_kwargs: dict = {}) -> None:
        self.organs_labels_dict = organs_labels_dict
        self.metrics_kwargs = metrics_kwargs

    def execute(self, fpath_gt, fpath_pred):
        self.surface_distances = None
        self.case_results_list = []
        img_gt = sitk.ReadImage(fpath_gt)
        img_gt_np = sitk.GetArrayFromImage(img_gt)
        img_pred = sitk.ReadImage(fpath_pred)
        img_pred_np = sitk.GetArrayFromImage(img_pred)

        assert all(
            np.array(img_gt.GetSpacing()) == np.array(img_pred.GetSpacing())
        ), "spacing does not match"
        self.spacing_np = img_gt.GetSpacing()[::-1]

        for organ_name, organ_label in self.organs_labels_dict.items():
            if organ_name == "background":
                continue
            self.organ_info_dict = {
                "organ_name": organ_name,
                "organ_label": organ_label,
            }
            self.img_gt_np_bool = img_gt_np == organ_label
            self.img_pred_np_bool = img_pred_np == organ_label

            self.check_if_missing()

            self.compute_all(
                hd_percentile=self.metrics_kwargs.get("hd_percentile"),
                func=self.metrics_kwargs.get("asd_function"),
                tolerance_mm=self.metrics_kwargs.get("surface_dice_tolerance"),
            )

        return self.case_results_list

    def check_if_missing(self):
        if self.img_gt_np_bool.astype(int).sum() == 0:
            missing = 1
        else:
            missing = 0
        self.organ_info_dict['missing_organ_on_gt'] = missing

        if self.img_pred_np_bool.astype(int).sum() == 0:
            missing = 1
        else:
            missing = 0
        self.organ_info_dict['missing_organ_on_pred'] = missing


    def compute_all(
        self, **kwargs,
    ):
        # hd_percentile=None, asd_function=None, surface_dice_tolerance=None
        self.compute_vol_dice()
        self.compute_surface_distances()
        self.compute_assd()
        self.compute_hausdorff(percentile=kwargs.get("hd_percentile"))
        self.compute_average_surface_distance(func=kwargs.get("asd_function"))
        self.compute_surface_dice_at_tolerance(
            tolerance_mm=kwargs.get("surface_dice_tolerance")
        )

    def compute_vol_dice(self):
        metric_name = "volumetric_dice"
        value = metrics.compute_dice_coefficient(
            mask_gt=self.img_gt_np_bool, mask_pred=self.img_pred_np_bool
        )
        self.save_result(metric=metric_name, value=value)

    def compute_surface_distances(self):
        self.surface_distances = metrics.compute_surface_distances(
            mask_gt=self.img_gt_np_bool,
            mask_pred=self.img_pred_np_bool,
            spacing_mm=self.spacing_np,
        )

    def compute_assd(self):
        metric_name = "assd"
        value = metrics.compute_assd(self.surface_distances)
        self.save_result(metric=metric_name, value=value)

    def compute_hausdorff(self, percentile=None):
        metric_name = "hausdorff"
        if percentile is None:
            percentile = 95
        value = metrics.compute_robust_hausdorff(self.surface_distances, percentile)

        self.save_result(
            metric=metric_name, value=value, parameters_str=f"percentile={percentile}",
        )

    def compute_average_surface_distance(self, func=None):
        metric_name = "average_surface_distance"
        if func is None:
            func = np.mean
        value = metrics.compute_average_surface_distance(self.surface_distances)

        self.save_result(
            metric=metric_name,
            value=func(value),
            parameters_str=f"function={func.__name__}",
        )

    def compute_surface_dice_at_tolerance(self, tolerance_mm=None):
        metric_name = "surface_dice_at_tolerance"
        if tolerance_mm is None:
            tolerance_mm = 2.0
        value = metrics.compute_surface_dice_at_tolerance(
            self.surface_distances, tolerance_mm=tolerance_mm
        )

        self.save_result(
            metric=metric_name,
            value=value,
            parameters_str=f"tolerance_mm={tolerance_mm}",
        )

    def save_result(self, metric, value, parameters_str="no_parameters"):
        self.case_results_list.append(
            {
                **self.organ_info_dict,
                **{"metric": metric, "value": value, "parameters": parameters_str,},
            }
        )


# def compute_metrices_deepmind(y, y_pred, filename, phase, current_fold, spacing):
#     y = y.detach().cpu().numpy()[0]
#     y_pred = y_pred.detach().cpu().numpy()[0]

#     meandice = []
#     hausdorff_distance95 = []
#     average_surface_distance = []
#     surface_dice = []
#     assd = []
#     # compute metrics for all labels, but for the background label
#     for i in range(1, y.shape[0]):
#         y_bool_gt = y[i].astype(bool)
#         y_bool_pred = y_pred[i].astype(bool)

#         assert isinstance(y_bool_gt, np.ndarray)
#         assert y_bool_gt.ndim == 3
#         assert isinstance(y_bool_pred, np.ndarray)
#         assert y_bool_pred.ndim == 3

#         tmp = metrics.compute_dice_coefficient(y_bool_gt, y_bool_pred)
#         meandice.append(tmp)

#         surface_distances = metrics.compute_surface_distances(
#             y_bool_gt, y_bool_pred, spacing_mm=spacing
#         )

#         tmp = metrics.compute_assd(surface_distances)
#         assd.append(tmp)

#         tmp = metrics.compute_robust_hausdorff(surface_distances, 95)
#         hausdorff_distance95.append(tmp)

#         tmp = metrics.compute_average_surface_distance(surface_distances)
#         average_surface_distance.append(np.mean(tmp))

#         tmp = metrics.compute_surface_dice_at_tolerance(
#             surface_distances, tolerance_mm=2.85
#         )
#         surface_dice.append(np.mean(tmp))

#     meandice_out = metric_to_dict2(
#         np.asarray(meandice, dtype=np.float),
#         filename=filename,
#         phase=phase,
#         metric="MeanDice",
#         include_background=False,
#         fold=current_fold,
#     )
#     assd_out = metric_to_dict2(
#         np.asarray(assd, dtype=np.float),
#         filename=filename,
#         phase=phase,
#         metric="ASSD",
#         include_background=False,
#         fold=current_fold,
#     )
#     hausdorff_distance95_out = metric_to_dict2(
#         np.asarray(hausdorff_distance95, dtype=np.float),
#         filename=filename,
#         phase=phase,
#         metric="HD95",
#         include_background=False,
#         fold=current_fold,
#     )
#     average_surface_distance_out = metric_to_dict2(
#         np.asarray(average_surface_distance, dtype=np.float),
#         filename=filename,
#         phase=phase,
#         metric="AvgSurfaceDist",
#         include_background=False,
#         fold=current_fold,
#     )
#     surface_dice_out = metric_to_dict2(
#         np.asarray(surface_dice, dtype=np.float),
#         filename=filename,
#         phase=phase,
#         metric="SurfaceDice",
#         include_background=False,
#         fold=current_fold,
#     )

#     # assd = monai.metrics.compute_average_surface_distance(
#     #     y_pred,
#     #     y,
#     #     include_background=False,
#     #     symmetric=True,
#     #     distance_metric="euclidean",
#     # )
#     # assd_out = metric_to_dict2(
#     #     assd * multiply_factor,
#     #     filename=filename,
#     #     phase=phase,
#     #     metric="ASSD",
#     #     include_background=False,
#     #     fold=current_fold,
#     # )

#     return (
#         meandice_out
#         + hausdorff_distance95_out
#         + average_surface_distance_out
#         + surface_dice_out
#         + assd_out
#     )
