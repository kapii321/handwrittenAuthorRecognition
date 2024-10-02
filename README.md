# Author detection based on handwritten text style
- Data given are scanned/photographed pages of handwritten text written by 8 distinct authors

# Code flow
- First divide page scans into separate words and save them into a folder, one folder for each author(AllWords/Wauthor#)
- Next, normalize images to the same size by pasting them onto a white rectangle normBackground.bmp

<img width="438" alt="PageOfWords" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/PanTadeusz_1.bmp">

<img width="438" alt="SingleWord" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/1.bmp">

- Next, if needed rescale the images to smaller size so training requires less computing power
- Next, train and test the model
- Preview results
<img width="438" alt="Result" src="https://github.com/kapii321/aiProjectNew/blob/a698a87afb4bae00a23197675cedf2216da8c1ce/docImg/result.png">
