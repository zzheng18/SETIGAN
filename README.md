# SETIGAN

The notebook Generate data.ipynb contains all the code you need to generate the training data. 

The folder Normalized_cGAN contains the files you need to train the network and do classification using the discriminator. To run conditional_DCGAN.py, you will also need a list of file names and labels, which can either be generated in Generate data.ipynb or can be obtained by this link: https://drive.google.com/drive/folders/1fez_gTHDoFQVMG4FtL7vsRO7o6nkOXcN?usp=sharing

You can either train the model from scratch or use the trained model avaliable in the link above. If you do not want the normalization of the output of classifier when training, remember to delete the nn.Sigmoid() layer at the end of the classifier. 
