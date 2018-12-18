import matplotlib.pyplot as plt

def prediction(fn):
    root = 'data/prediction_images/'
    img = plt.imread(root+fn)
    plt.imshow(img);


if __name__ == '__main__':
    print('test')
    img = plt.imread('data/prediction_images/boxer_test.jpeg')
    plt.imshow(img)
    print('test')
