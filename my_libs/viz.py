import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools

output_notebook()


def plot_image(fp, w=5, h=5):
    plt.figure(figsize=(w, h))
    plt.imshow(Image.open(fp))

def plot_images(list_fps, ncols, nrows, w, h):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h))
    ax = ax.ravel()
    for idx, fp in enumerate(list_fps):
        ax[idx].imshow(Image.open(fp))
    
    plt.show()


def plot_dimensions(list_fps, h=500, w=300):
    list_w, list_h = [], []

    for p in list_fps:
        im = np.asarray(Image.open(p))
        list_w.append(im.shape[1])
        list_h.append(im.shape[0])
    
    # plotting width distribution
    hist, edges = np.histogram(list_w)
    mean, median, min_, max_ = np.mean(list_w), np.median(list_w), np.min(list_w), np.max(list_w)
    p1 = figure(title=f"Image width distribution  - mean:{mean:.0f} - median:{median:.0f} - min:{min_:.0f} - max:{max_:.0f}",
      background_fill_color="#fafafa", tools="hover")
    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="blue", line_color="white", alpha=0.5)

    # plotting height distribution
    hist, edges = np.histogram(list_h)
    mean, median, min_, max_ = np.mean(list_h), np.median(list_h), np.min(list_h), np.max(list_h)
    p2 = figure(title=f"Image height distribution  - mean:{mean:.0f} - median:{median:.0f} - min:{min_:.0f} - max:{max_:.0f}",
      background_fill_color="#fafafa", tools="hover",)
    p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="blue", line_color="white", alpha=0.5)

    show(gridplot([p1, p2], ncols=2, plot_height=h, plot_width=w))


def plot_confusion_matrix(y_true, y_pred, title='Confusion matrix'):
    """
    """

    # print classification report
    print(classification_report(y_true, y_pred))

    # plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # tick_marks = np.arange(len(target_names))
    # plt.xticks(tick_marks, target_names, rotation=45)
    # plt.yticks(tick_marks, target_names)
    tick_marks = [-0.5] + list(np.arange(len(np.unique(y_true)))) + [len(np.unique(y_true))-0.5]
    tick_labs = [""] + [str(i) for i in list(np.arange(len(np.unique(y_true))))] + [""]
    plt.xticks(tick_marks, tick_labs, rotation=45)
    plt.yticks(tick_marks, tick_labs)
    

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.show()
