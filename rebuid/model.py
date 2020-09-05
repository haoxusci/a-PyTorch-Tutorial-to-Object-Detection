from torch import nn
import torch.nn.functional as F
import torchvision
from utils import *
import torch
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # changed from VGG16 for layer 5
        self.conv5_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # change from FC to Conv layer for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6,
        )
        self.conv7 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38)

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # low-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(
            pretrained=True
        ).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # transfer conv.parameters from pretrained model to current model
        # except conv6 and conv7 because of the change
        for i, param_name in enumerate(param_names[:-4]):
            state_dict[param_name] = pretrained_state_dict(
                pretrained_param_names[i]
            )

        # convert for conv6 and conv7
        conv_fc6_weight = pretrained_state_dict["classifier.0.weight"].view(
            4096, 512, 7, 7
        )  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict["classifier.0.bias"]  # (4096)
        conv_fc7_weight = pretrained_state_dict["classifier.3.weight"].view(
            4096, 4096, 1, 1
        )  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict["classifier.3.bias"]  # (4096)

        state_dict["conv6.weight"] = decimate(
            conv_fc6_weight, m=[4, None, 3, 3]
        )  # (1024, 512, 3, 3)
        state_dict["conv6.bias"] = decimate(conv_fc6_bias, m=[4])  # (1024)

        state_dict["conv7.weight"] = decimate(
            conv_fc7_weight, m=[4, 4, None, None]
        )  # (1024, 1024, 1, 1)
        state_dict["conv7.bias"] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # is this necessary? /hao
        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv8_1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv8_2 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv9_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv9_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv10_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv10_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.conv11_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv11_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        out = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)
        conv11_2_feats = out  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        """initialize the predictions"""
        super().__init__()
        self.n_classes = n_classes

        # number of prior-boxes
        n_box = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4,
        }

        # localization prediction
        self.loc_conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=n_box["conv4_3"] * 4,
            kernel_size=3,
            padding=1,
        )
        self.loc_conv7 = nn.Conv2d(
            in_channels=1024,
            out_channels=n_box["conv7"] * 4,
            kernel_size=3,
            padding=1,
        )
        self.loc_conv8_2 = nn.Conv2d(
            in_channels=512,
            out_channels=n_box["conv8_2"] * 4,
            kernel_size=3,
            padding=1,
        )
        self.loc_conv9_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv9_2"] * 4,
            kernel_size=3,
            padding=1,
        )
        self.loc_conv10_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv10_2"] * 4,
            kernel_size=3,
            padding=1,
        )
        self.loc_conv11_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv11_2"] * 4,
            kernel_size=3,
            padding=1,
        )

        # class prediction
        self.cl_conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=n_box["conv4_3"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        self.cl_conv7 = nn.Conv2d(
            in_channels=1024,
            out_channels=n_box["conv7"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        self.cl_conv8_2 = nn.Conv2d(
            in_channels=512,
            out_channels=n_box["conv8_2"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        self.cl_conv9_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv9_2"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        self.cl_conv10_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv10_2"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        self.cl_conv11_2 = nn.Conv2d(
            in_channels=256,
            out_channels=n_box["conv11_2"] * n_classes,
            kernel_size=3,
            padding=1,
        )
        # Initialize convolutions' paramters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameter
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)

    def forward(
        self,
        conv4_3_feats,
        conv7_feats,
        conv8_2_feats,
        conv9_2_feats,
        conv10_2_feats,
        conv11_2_feats,
    ):
        """
        forward function
        """
        batch_size = conv4_3_feats.szie(0)
        # for location prediction
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        # this is to ensure it is stored in a contiguous chunk of memeory, # need for .vew()
        l_conv4_3 = l_conv4_3.permute(
            0, 2, 3, 1
        ).contiguous()  # (N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        # for class prediction
        c_conv4_3 = self.cl_conv4_3(
            conv4_3_feats
        )  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8_2 = self.cl_conv8_2(
            conv8_2_feats
        )  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv10_2(
            conv10_2_feats
        )  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)

        c_conv11_2 = self.cl_conv11_2(
            conv11_2_feats
        )  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)

        # A totoal of 8732 boxes for SSD300
        locs = torch.cat(
            [l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2],
            dim=1,
        )
        classes_scores = torch.cat(
            [c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
            dim=1,
        )

        return locs, classes_scores


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # since lower leverl features (conv4_3_feats) have considerably larger
        # scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each
        # channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        # (N, 512, 38, 38), (N, 1024, 19, 19)
        conv4_3_feats, conv7_feats = self.base(image)
        norm = (
            (conv4_3_feats.pow(2)).sum(dim=1, keepdim=True).sqrt()
        )  # L2 norm
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        # Pytorch autobroadcasts singleton dimensions during arithmetic
        # (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        (
            conv8_2_feats,
            conv9_2_feats,
            conv10_2_feats,
            conv11_2_feats,
        ) = self.aux_convs(conv7_feats)

        locs, classes_scores = self.pred_convs(
            conv4_3_feats,
            conv7_feats,
            conv8_2_feats,
            conv9_2_feats,
            conv10_2_feats,
            conv11_2_feats,
        )
        # (N, 8732, 4), (N, 8732, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        create the 8732 prior (default) boxes for the SSD300, defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of (8732, 4)
        """
        fmap_dims = {
            "conv4_3": 38,
            "conv7": 19,
            "conv8_2": 10,
            "conv9_2": 5,
            "conv10_2": 3,
            "conv11_2": 1,
        }
        obj_scales = {
            "conv4_3": 0.1,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
            "conv11_2": 0.9,
        }
        aspect_ratios = {
            "conv4_3": [1.0, 2.0, 0.5],
            "conv7": [1.0, 2.0, 0.5, 3.0, 0.333],
            "conv8_2": [1.0, 2.0, 0.5, 3.0, 0.333],
            "conv9_2": [1.0, 2.0, 0.5, 3.0, 0.333],
            "conv10_2": [1.0, 2.0, 0.5],
            "conv11_2": [1.0, 2.0, 0.5],
        }
        fmaps = list(fmap_dims.keys())
        prior_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append(
                            [
                                cx,
                                cy,
                                obj_scales[fmap] * sqrt(ratio),
                                obj_scales[fmap] / sqrt(ratio),
                            ]
                        )
                        # for ratio of 1, add an additional scale
                        if ratio == 1.0:
                            try:
                                additional_scale = sqrt(
                                    obj_scales[fmap] * obj_scales[fmaps[k + 1]]
                                )
                            # for the last feature map, there is no "next"
                            # feature map
                            except IndexError:
                                additional_scale = 1.0
                            prior_boxes.append(
                                [cx, cy, additional_scale, additional_scale]
                            )
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        # this does not ensure the box exceed the image yet.
        prior_boxes.clamp_(0, 1)
        # (8732, 4)
        return prior_boxes

    def detect_objects(
        self, predicted_locs, predicted_scores, min_score, max_overlap, top_k
    ):
        """
        Decipherthe 8732 locations and class scores (output of the SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        Parameters:
            predicted_locs (Tensor):
                predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimension (N, 8732, 4)
            predicted_scores (Tensor):
                class scores for each of the encoded locations/boxes, a tensor of dimension (N, 8732, n_classes)
            min_score:
                minimum threhold for a box to be consiered a match for a certain class
            max_overlap:
                maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'

        Returns:
            dtections (boxes, labels, scores). lists of length batch_size
        """
        batch_size = predicted_locs(0)  # N
        n_priors = self.priors_cxcy.size(0)  # 8732
        predicted_scores = F.softmax(predicted_scores, dim=2)
        # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels and scores forall images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # in (minx, miny, maxx, maxy) format
            decoded_locs = cxcy_to_xy(
                gcxgcy_cxcy(predicted_locs[i], self.priors_cxcy)
            )  # (8732, 4), fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # this line is useless thus masked out
            # max_scores, best_label = predicted_scores[i].max(dim=1)
            # (8732), (8732)
            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.bool
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[
                    score_above_min_score
                ]  # (n_qualified), < 8732
                # should be pytorch autobroadcast here
                class_decoded_locs = class_scores[score_above_min_score]
                # (n_qualified, 4)

                # sort predicted boxes adn scores by scores, decending
                class_scores, sort_ind = class_scores.sort(
                    dim=0, descending=True
                )  # (n_qualified), (n_qualified)
                class_decoded_locs = class_decoded_locs[
                    sort_ind
                ]  # (n_qualified, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(
                    class_decoded_locs, class_decoded_locs
                )  # (n_qualified, n_qualified)

                # Non-Maximum Suppression (NMS)
                # set a torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros(
                    (n_above_min_score), dtype=torch.uint8
                ).to(
                    device
                )  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs):
                    # leave it if already suppressed
                    if suppress[box] == 1:
                        continue
                    overlap_above_max = torch.tensor(
                        overlap[box] > max_overlap,
                        dtype=suppress.dtype,
                        device=suppress.device,
                    )
                    suppress = torch.max(suppress, overlap_above_max)
                    # dont support itself, even though an overlap of 1
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes(class_decoded_locs[1 - suppress])
                image_label.append(
                    torch.LongTensor((1 - suppress).sum().item() * [c]).to(
                        device
                    )
                )
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(
                    torch.FloatTensor([[0.0, 0.0, 1.0, 1.0]]).to(device)
                )
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.0]).to(device))
            # concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(
                    dim=0, descending=True
                )
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return (
            all_images_boxes,
            all_images_labels,
            all_images_scores,
        )  # lists of length batch_size, and each list corresponds to an image
