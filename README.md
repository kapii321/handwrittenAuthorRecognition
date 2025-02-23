# Author Detection Based on Handwritten Text Style

A PyTorch-based deep learning project that identifies authors from their handwriting style. The model processes scanned/photographed pages of handwritten text from 8 distinct authors.

# Code flow
- First divide page scans into separate words and save them into a folder, one folder for each author(AllWords/Wauthor#)
<img width="438" alt="PageOfWords" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/PanTadeusz_1.bmp">

- Next, normalize images to the same size by pasting them onto a white rectangle normBackground.bmp
<img width="438" alt="SingleWord" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/1.bmp">

- Next, if needed rescale the images to smaller size so training requires less computing power
- Next, train and test the model:
  - Dataset split: 80% training, 10% validation, 10% testing
  - CNN with 2 convolutional layers
  - Adam optimizer (lr=0.001)
  - 20 epochs
  - Batch size of 64

- Preview results
<img width="438" alt="Result" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/result.png">


**Technologies Used**

- Python
- PyTorch
- OpenCV
- Matplotlib
