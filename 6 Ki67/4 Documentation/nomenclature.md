# Model names

**BRAP-Net** combines frozen Phikon-v2 image features with the gain-trained connectome for joint BCData + SHIDC-B-Ki-67 three-class classification. The anatomical graph and the degree-preserving rewired graph are two variants. Each graph and optimization seed has its own selected weights; the same checkpoint predicts both dataset subsets.

DINOv2 is the alternative frozen extractor. Separately trained models are named by extractor and graph, such as **Phikon-v2 + anatomical** or **DINOv2 + rewired**. Native RGB supplies pixel values directly to the connectome, without a ViT.

Internal identifiers remain `dinov2`, `phikonv2`, `anatomical` and `degree241`. The rewiring seed is 241; optimization seeds are 1729, 2718 and 3141. These are different experimental roles.

Checkpoint and extractor selection are defined in the [joint protocol](joint_training.md).
