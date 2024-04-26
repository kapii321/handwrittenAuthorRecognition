# A program for reading and displaying handwritten words downloaded from graphic 
# files based on descriptions from text files
from matplotlib import image as mpimg
from PIL import Image
import os

num_of_authors = 8
file_nums = [10, 7, 10, 4, 8, 7, 7, 10]
max_num_of_words_per_author = 0


def findMaxPhoto():
    max_dims = [0, 0]
    for author_no in range(num_of_authors):
        file_desc_name = "author" + str(author_no + 1) + "/word_places.txt"
        file_desc_ptr = open(file_desc_name, 'r')
        text = file_desc_ptr.read()
        lines = text.split('\n')
        number_of_lines = lines.__len__() - 1
        row_values = lines[0].split()
        num_of_words = 0
        image_file_name_prev = ""

        for i in range(number_of_lines):
            row_values = lines[i].split()
            if row_values[0] != '%':
                image_file_name = "author" + str(author_no + 1) + "\\" + row_values[0][1:-1]
                # pdb.set_trace()
                if image_file_name != image_file_name_prev:
                    image_file_name_prev = image_file_name
                row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                    int(row_values[4]), int(row_values[5])
                if (row2 - row1 >= max_dims[0]):
                    max_dims[1] = row2 - row1
                if (column2 - column1 >= max_dims[1]):
                    max_dims[0] = column2 - column1
                num_of_words += 1
                # pdb.set_trace()
            if num_of_words >= max_num_of_words_per_author:
                break

        file_desc_ptr.close()
    return max_dims


def divideAndSaveWords():
    for author_no in range(num_of_authors):
        file_desc_name = "author" + str(author_no + 1) + "/word_places.txt"
        file_desc_ptr = open(file_desc_name, 'r')
        text = file_desc_ptr.read()
        lines = text.split('\n')
        number_of_lines = lines.__len__() - 1
        row_values = lines[0].split()
        number_of_values = row_values.__len__()
        max_num_of_words_per_author = number_of_lines-2

        num_of_words = 0
        image_file_name_prev = ""
        for i in range(number_of_lines):

            row_values = lines[i].split()

            if row_values[0] != '%':
                number_of_values = len(row_values)
                image_file_name = "author" + str(author_no + 1) + "\\" + row_values[0][1:-1]
                # pdb.set_trace()
                if image_file_name != image_file_name_prev:
                    image = mpimg.imread(str(image_file_name))
                    image_file_name_prev = image_file_name
                word = row_values[1]
                row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                    int(row_values[4]), int(row_values[5])

                subimage = image[row1:row2, column1:column2]
                img = Image.open("normBackground.bmp")
                img2 = Image.fromarray(subimage)
                img.paste(img2, (2, 10))
                if word[0] == '<' or word[-1] == '>' or word[0] == "\"" or word[-1] == "\"":
                    word = word[1:-2]
                elif word == "||":
                    word = "ddW"
                elif word[-1] == '?':
                    word = word[:-1]

                filename = f"./AllWords/Wauthor{str(author_no + 1)}/{str(num_of_words)}.bmp"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                img.save(filename)

                # plt.title("Author " + str(author_no + 1) + ", image = " + row_values[0][1:-1] + ", word = " + word)
                # plt.xlabel("X")
                # plt.ylabel("Y")
                # plt.imshow(subimage)
                # plt.imshow(img)
                # plt.show()
                num_of_words += 1
                # pdb.set_trace()
            if num_of_words >= max_num_of_words_per_author:
                break

        file_desc_ptr.close()

divideAndSaveWords()
