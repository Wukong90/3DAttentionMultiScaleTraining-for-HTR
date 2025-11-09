**Read this in other languages: [English](README.md),[中文](README_zh.md).**

# Three-dimensional attention multi-scale training network (TDMTNet)

In the handwritten text recognition task, the HMM has a higher modeling resolution for each character by using mutiple states. However, it needs a complex training and inference pipeline, including the feature design, the generative model GMM-HMM, the discriminative  model NN-HMM and the language model. The CTC and the ED methods use more efficient and direct mathematic computation and the corresponding networks can be trained in an end-to-end way. Compared with the HMM, their modeling resolutions of characters are lower. The recognition networks are prone to attention drift when applied to long handwritten text recognition. Previous experience indicates that they need more traing text.

Inspired by the typical segmentation-free approaches (HMM、CTC and Encoder-Decoder), we design a neural network that can absorb the advantages of the three segmentation
-free methods. The structure of the TDMTNet is shown in the following figure and has four characteristics:

(1) The CNN with the hybrid attention module (HAM);

(2) The 3D attention module;

(3) The features fusion module;

(4) The multi-scale training.

![](https://github.com/Wukong90/3DAttentionMultiScaleTraining-for-HTR/blob/main/imgs/diff_methods.png)

The following figure shows the proposed the muti-scale training and the parts within the dashed lines are only used during the training stage. It includes two parts. Firstly, in the training stage, multiple feature sequences with different frame lengths are extracted. Although parallel branches including the same 3D attention block and global-local context block are used, we only retained the corresponding branch of the frame length 3 during the inference stage in our experiments. Secondly, we use the joint training of the CTC and the CE losses. 

![](https://github.com/Wukong90/3DAttentionMultiScaleTraining-for-HTR/blob/main/imgs/net_structure.png)

# Codes and trained weights

The TDMTNet code for handwritten text line recognition and the training and testing codes have been released (the cross entropy loss based fine-tuning code has not been released yet). The training/testing code for the Chinese dataset is train_TDMSNet_Chinese.py, and the training/testing code for the English dataset is train_TDMSNet_eng.py. The network model is located in model/model.py. The directory configure contains the main configuration files, parameter settings, dataset construction and image preprocessing codes. The directory Datasets_list is used to store training/test image data and file-name lists. The directory weights contains the trained network weights by using different datasets. A weight name only containing CTC indicates that the weight has not been fine-tuned by the CE loss while a weight name including CTC_CE indicates that the network weight has been fine-tuned by the CE loss. All weights contain three complete branches. In the inference stage, we actually only need to keep the branch built by the window length of 3.

# Experimental datasets

The proposed network is validated on two latest Chinese handwritten text datasets (SCUT-HCCDoc and SCUT-EPT) and an important English handwritten text dataset IAM.

In the SCUT-HCCDoc dataset, the original training set contains 93,254 text line images, and 1,993 low-quality (illegible text, random graffiti, high overlap of text and background, and incomplete characters) or vertical writing text were deleted. Therefore, only 91,261 text line images were used for network training. The train_list.txt and test_list.txt in the directory Datasets_list/SCUT-HCCDoc/ are the training image list and test image list we used in our experiments. The test set contains all original test images of the dataset. It should be noted that the creator of the dataset only provides page-level text images and the contained text line annotations (json files). Our lists include the names of text line images. The naming method of a text line is its' corresponding original page-level image name followed by the corresponding line number. A line number is the order of the corresponding text line in the original annotation json file. The directory Datasets_list/SCUT-HCCDoc/abnormal_lists_and_images contains the low-quality or vertically written text images in the original training set, which were excluded. All abnormal images are located in the abnormals directory, all_abnormal_list.txt is a list of all abnormal images, *_abnormal.txt is a list of abnormal text images in the corresponding subset, and * is the name of the original data subset.

The files TrainDataRuChar2Int_HCCDoc.npy and TrainDataRuInt2Char_HCCDoc.npy under the directory Datasets_list/SCUT-HCCDoc store the corresponding relationship between the characters and network output nodes.

In the SCUT-EPT dataset, 681 abnormal handwritten texts including swapping or overlapped characters were deleted, and only 39,319 text line images were used for training. The train_list.txt and the test_list.txt in Datasets_list/SCUT-EPT/ are the training image list and the test image list we used in our experiments. The test set contains all original test images of the dataset. The file abnormal.txt is a list including 681 abnormal images that were excluded from the original training set. About their detailed information, please refer to our project https://github.com/Wukong90/EHT-Dataset.SCUT-EPT-Abnormal.

The files TrainDataRuChar2Int_EPT.npy and TrainDataRuInt2Char_EPT.npy under the directory Datasets_list/SCUT-EPT/ store the corresponding relationship between the characters and network output nodes.

The standard IAM dataset provides a training set, two validation sets, and a test set. The trainset.txt, validationset1.txt, validationset2.txt, and testset.txt listed in the directory Datasets_list/IAM/split/ are the corresponding training list, validation lists, and test list. In our experiments, all original training iamges were used for training, all images in two validation sets were used to select the best model and all original images in the test set were used to evaluate the final performance of the model.

Two latest challenging Chinese handwritten datasets SCUT-HCCDoc and SCUT-EPT can be obtained from https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release and https://github.com/HCIILAB/SCUT-EPT_Dataset_Release?tab=readme-ov-file respectively. The English handwritten dataset IAM can be obtained from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. It should be noted that since 2018, most related researchers have adopted the so-called RWTH data partition, which is different from the standard IAM dataset partition.

 



