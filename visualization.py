import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from embeddings.route_embeddings import *
from embeddings.embedding_helpers import *

# TODO: 
# - add support for different difficulty scales, e.g. Font-Scale, V-Grade

def bar_plot(df, col='font_scale'): 
    '''
    Bar plot of the number of routes for each grade ('6A+' to '8B+')

        Parameters:
            col (str) : can be 'font_scale', 'v_grade' or 'user_rating'
            df (pandas.core.frame.DataFrame) : Dataframe, must contain 'col' as a Column 
    '''
    df[col].value_counts().sort_index().plot(kind='barh')
    return

# Equality Condition on conditional_col and conditional_val, e.g. select all routes where Grade == 8B+
# For the values in 'col', this plots the count of each unique element. E.g. For each user_rating, plot how many routes there are 
# (within grade 8B+).
def condtional_bar_plot(df, col='user_rating', condition_col='Grade', condition_val='8B+'):
    df[df[condition_col] == condition_val][col].value_counts().sort_index().plot(kind='barh')
    return

# Bar Plot with 4 bars for each grade ('6A+' to '8B+'), the 4 bars show the number of routes with grade 0,1,2 or 3. 
def plot_user_rating_per_grade(df, scale = 'font_scale'):
    assert scale == 'font_scale' or scale == 'v_grade'

    df_rating = pd.DataFrame({
            'userrating0': df[df['user_rating'] == 0][scale].value_counts(),
            'userrating1': df[df['user_rating'] == 1][scale].value_counts(),
            'userrating2': df[df['user_rating'] == 2][scale].value_counts(),
            'userrating3': df[df['user_rating'] == 3][scale].value_counts(),
        }, columns=['userrating0', 'userrating1', 'userrating2', 'userrating3'])
    df_rating[scale] = df_rating.index
    df_rating = df_rating.fillna(0)
    df_rating.plot(x=scale, kind='barh', stacked=False)
    return

def route_count_grouped_by(df, groupby_index='font_scale'):
    user_rating_per_grade = df.groupby([groupby_index]).agg({'user_rating': [len]})['user_rating']

    x = user_rating_per_grade.index
    y = user_rating_per_grade['len']

    plt.bar(x, y)
    plt.xlabel(f'{groupby_index}')
    plt.ylabel('Number of Routes')
    plt.title(f'Number of Routes per {groupby_index}')
    plt.show()
    return

def plot_user_rating_per_grade_w_error_bar(df, groupby_index='font_scale'):
    user_rating_per_grade = df.groupby([groupby_index]).agg({'user_rating': [np.mean, np.std]})['user_rating']

    x = user_rating_per_grade.index
    y = user_rating_per_grade['mean']
    err = user_rating_per_grade['std']

    plt.errorbar(x, y, err, linestyle=None, marker='^')
    plt.xlabel(f'{groupby_index}')
    plt.ylim(0, 3)
    plt.ylabel('Average User Rating')
    plt.title(f'Average User Rating grouped by {groupby_index}')
    plt.show()
    return

# takes a Route represented as a list of strings: e.g. ['B5', 'G5', 'D7', 'G9', 'B12', 'G15', 'C18']
# creates a plot of the route as a 11 times 18 grid, Holds that are part of the route are dark green.
def plot_route(route, title = None):
    m = bag_of_holds([route])
    m = reshape_1d_to_2d(m)[0]

    plt.imshow(m, cmap = 'Greens', origin = 'lower')
    plt.xticks(list(range(config.N_COLS)), labels = [chr(65 + i) for i in range(config.N_COLS)])
    plt.yticks(list(range(config.N_ROWS)), labels = list(range(1, config.N_ROWS + 1)))
    if title is not None:
        plt.title(label = title)
    plt.show()
    return

# takes a Dataframe and the index. Plots the route from that index.
def plot_route_from_df(df, route_i, scale='font_scale'):
    assert scale == 'font_scale' or scale == 'v_grade'

    route = df['holds'][route_i]
    grade = df[scale][route_i] 
    plot_route(route, f'Route of Grade {grade}')
    return


# Plot a heatmap (11 times 18 grid) of the holds used for routes of that grade. Darker (green) values
# mean that a hold is frequently used in routes of that grad. Bright values mean it is hardly used.
# grade:     'all', '6A+', '6B+', '7B+', '7A', '7A+', '6C+', '6C', '6B', '7C', '7C+',
#            '7B', '8B', '8A', '8B+', or '8A+'.
def plot_grade_heatmap(df, grade, scale='font_scale'): 
    assert scale == 'font_scale' or scale == 'v_grade'

    heatmap = np.zeros((1, config.N_ROWS * config.N_COLS))
    if grade == 'all':
        for holds in df.holds:
            try:
                heatmap[0] += bag_of_holds([holds])[0]
            except:
                print(f'Could not build bag-of-holds embeddings for: {holds}')

    else:
        for holds in df[df[scale] == grade].holds:
            try:
                heatmap[0] += bag_of_holds([holds])[0]
            except:
                print(f'Could not build bag-of-holds embeddings for: {holds}')

    
    heatmap = reshape_1d_to_2d(heatmap)[0]

    plt.imshow(heatmap, cmap = 'Greens', origin = 'lower')
    plt.xticks(list(range(config.N_COLS)), labels = [chr(65 + i) for i in range(config.N_COLS)])
    plt.yticks(list(range(config.N_ROWS)), labels = list(range(1, config.N_ROWS + 1)))
    plt.title(label=f'Heatmap of Holds used in Routes of Grade {grade}')
    plt.show()
    return

# given the predicted values (y_pred) and the ground trouth, plots a confusion matrix 
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

# plots a line chart of a routes probability as beeing classified of a certain grade.
# y_true:      Ground trouth, e.g. 4
# y_probas:    Classifiers Probabilities of predicting a certain class ( .predict_proba() in sklearn)
#              e.g.[0.03515419 0.12533069 0.41219004 0.14129291 0.12183386 0.12676201
#              0.01930746 0.00851937 0.00437948 0.00145201 0.00377798]
def plot_classifiers_certainty(y_true, y_probas, show_full_y_axis=False) -> None:
    from data_loading import V_GRADE
    import matplotlib.ticker as mtick
    plt.plot(y_probas[0], label='Predicted Rating')
    plt.axvline(x=y_true, color='green', label='Given Rating')

    plt.xlabel("V Grade")
    plt.xticks(list(V_GRADE.values()), list(V_GRADE.keys()))
    
    plt.ylabel("Probability")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    
    plt.title("Classifier's Certainty of the Route's Difficulty")
    
    if show_full_y_axis:
        plt.ylim(0,1)
        
    plt.legend()
    plt.show()
    return 