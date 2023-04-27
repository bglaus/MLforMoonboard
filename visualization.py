import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from embeddings import one_hot_embeddings


def bar_plot(df, col='Grade'):  # col can be 'Grade' or 'UserRating'
    df[col].value_counts().sort_index().plot(kind='barh')
    return

# plot bar plot of number of rows from each value from col, where the value in condition_col of the row is equal to condition_val
# e.g. plot UserRating filtered by Grade being equal to '8B+'.
def condtional_bar_plot(df, col='UserRating', condition_col='Grade', condition_val='8B+'):
    df[df[condition_col] == condition_val][col].value_counts().sort_index().plot(kind='barh')
    return

def plot_user_rating_per_grade(df):
    df_rating = pd.DataFrame({
            'userrating0': df[df['UserRating'] == 0]['Grade'].value_counts(),
            'userrating1': df[df['UserRating'] == 1]['Grade'].value_counts(),
            'userrating2': df[df['UserRating'] == 2]['Grade'].value_counts(),
            'userrating3': df[df['UserRating'] == 3]['Grade'].value_counts(),
        }, columns=['userrating0', 'userrating1', 'userrating2', 'userrating3'])
    df_rating['Grade'] = df_rating.index
    df_rating = df_rating.fillna(0)
    df_rating.plot(x='Grade',kind='barh',stacked=False)
    return


def plot_route(df, route_i):
    m = one_hot_embeddings.bitmap_2d([df['Moves'][route_i]])[0]
    grade = df['Grade'][route_i]
    plt.imshow(m, cmap ='Greens', origin='lower')
    plt.xticks(list(range(11)), labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
    plt.yticks(list(range(18)), labels=list(range(1,19)))
    plt.title(label=f'Route of Grade {grade}')
    plt.show()
    return

def plot_grade_heatmap(df, grade):
    heatmap = np.zeros((18, 11))
    if grade == 'all':
        for moves in df.Moves:
            heatmap += one_hot_embeddings.bitmap_2d([moves])[0]

    else:
        for moves in df[df.Grade == grade].Moves:
            heatmap += one_hot_embeddings.bitmap_2d([moves])[0]
    plt.imshow(heatmap, cmap ='Greens', origin='lower')
    plt.xticks(list(range(11)), labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
    plt.yticks(list(range(18)), labels=list(range(1,19)))
    plt.title(label=f'Heatmap of Holds used in Routes of Grade {grade}')
    plt.show()
    return

def plot_confusion_matrix(y_true, y_pred, normalize='true') -> None:
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion matrix')
    cbar = fig.colorbar(im)             # add a color bar
    plt.show()
    return