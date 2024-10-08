import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.manifold import TSNE
import math
import copy
import os
from scratch.utils.dr_metrics import compute_quality_criteria, compute_coranking_matrix


def generate_clusters(reducer : TSNE, data_label : np.ndarray, supervised : bool = False) -> pd.DataFrame:
  """
  Generates clusters based on data and a dimensionality reduction method.

    Args:
      reducer : a dimensionality reducing class that contains "fit_transform" method, like TSNE or UMAP.
      data_label(np.ndarray) : a data array where targets are in the first column.
      supervised(boolean) : specifies if the dimensionality reduction method is used in its supervised form.

    Returns:
      pd.DataFrame : returns a dataframe containing label, x and y of data reduced to 2 dimensions

  """

  data = data_label.T[1:].T
  if supervised:
    embedding = reducer.fit_transform(data, y = data_label.T[0].T)
  else:
    embedding = reducer.fit_transform(data)
  result = pd.DataFrame(embedding, columns=["x", "y"])
  result["label"] = data_label.T[0].T
  return result

def calculate_centroids(data: pd.DataFrame, num_lines : int = 24) -> np.ndarray:
  """
  Calculates centroids based on data coordinates on the embedded domain

    Args:
      data(pd.DataFrame) : a DataFrame containing ["label"] as in activityID, and ["x", "y"] for data coordinates on the embedded domain
      num_lines(int) : the amount of lines in the centroids array, indicates the maximum of classID's

    Returns:
      np.ndarray : an array containing ["x", "y"] for the centroids coordinates.

  """
  centroids = np.zeros((num_lines + 1, 3))
  for i in range(data["x"].shape[0]):
    centroids[math.ceil(data["label"][i])][0] += data["x"][i]
    centroids[math.ceil(data["label"][i])][1] += data["y"][i]
    centroids[math.ceil(data["label"][i])][2] += 1
  centroids = centroids[~np.all(centroids == 0, axis=1)]
  for i in range(centroids.shape[0]):
    centroids[i][0] /= centroids[i][2]
    centroids[i][1] /= centroids[i][2]
  return centroids

def get_distance(a, b):
  return np.sum(np.square(np.subtract(a, b)))

def calculate_distances(centroids : np.ndarray) -> np.ndarray:
  """
  Calculates distances between classes based on centroids

    Args:
      centroids(np.ndarray) : an array where centroids[i][0:2] contains the centroid of class i as [x, y] in the embedded domain

    Returns:
      np.ndarray : a square array where distances[i][j] contains the distances between classes i and j centroids's

  """
  distances = np.zeros((centroids.shape[0], centroids.shape[0]))
  for i in range(centroids.shape[0]):
    for j in range(centroids.shape[0]):
      distances[i][j] = get_distance(centroids[i][0:2], centroids[j][0:2])
  return distances

def calculate_similarities(distances : np.ndarray) -> np.ndarray:
  """
  Calculates similarities based on distances, so that it can be normalized to the [0, 1) interval

    Args:
      distances(np.ndarray) : a square array where distances[i][j] contains the distances between classes i and j centroids's

    Returns:
      np.ndarray : a square array where similarity[i][j] contains the similarity between classes i and j

  """
  similarities = np.zeros((distances.shape[0], distances.shape[0]))
  for i in range(distances.shape[0]):
    for j in range(distances.shape[0]):
      #using sigmoid clamping function
      similarities[i][j] = np.exp(-0.004 * distances[i][j])

  return similarities

def plot_similarity_matrix(reducer : TSNE, data_label : np.ndarray, label : dict, filepath : str, title : str):

  """
  Plots similarity matrix and saves it to filepath.

    Args:
      reducer(TSNE) : dimentionality reduction class.
      data_label(np.ndarray) : an array containing sensors data and the activity label in the first column.
      label(dict[int, str]): a dictionary mapping each activityID value to its activity.
      filepath(str) : where to save the matrix heatmap made.
      title(str) : name for the .png file generated.

  """
  max_classes = max(label.keys())
  embedding = generate_clusters(reducer, data_label, True)
  centroids = calculate_centroids(embedding, max_classes)
  distances = calculate_distances(centroids)
  similarities = calculate_similarities(distances)
  df_cm = pd.DataFrame(similarities, index = [i for i in label.values()],
                  columns = [i for i in label.values()])
  plt.figure(figsize = (16,16))
  sns.heatmap(df_cm, annot=True)
  plt.savefig(os.path.join(filepath  , title + ".png"))

def plot_scatter(df, figsize: tuple = (12, 12), title: str = None, labels: dict = None, filepath : str = None, filename : str = None, coeficients : dict = None):
    fig, ax = plt.subplots(figsize=figsize)
    for label, group_df in df.groupby("label"):
        label = labels[label] if labels is not None else label
        ax.scatter(group_df.x, group_df.y, label=label)
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
        colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]       
        for t,j1 in enumerate(ax.collections):
          j1.set_color(colorst[t])

    ax.legend()
    plt.suptitle(title)
    plt.title(f"Coefficients: Silhuette score: {coeficients['silhuette']:.4f}, Contiuity: {coeficients['continuity']:.4f}, Trustworthiness: {coeficients['trustworthiness']:.4f}, LCMC: {coeficients['lcmc']:.4f}")
    plt.savefig(os.path.join(filepath, filename + ".png"))

def plot_clustering(reducer, data_label, labels, title, supervised = False, is_original = True, original_data = None, filepath : str = None, filename : str = None):
  if(is_original):
    original_data = data_label.T[1:].T
  else:
    original_data = original_data.T[1:].T
  data = data_label.T[1:].T
  if supervised:
    embedding = reducer.fit_transform(data, y = data_label.T[0].T)
  else:
    embedding = reducer.fit_transform(data)
  result = pd.DataFrame(embedding, columns=["x", "y"])
  result["label"] = data_label.T[0].T
  coeficients = {}
  coeficients['silhuette'] = silhouette_score(embedding, result["label"])
  
  #Get metrics
  neighbors = np.unique(result['label']).shape[0]
  drcoef = compute_quality_criteria(compute_coranking_matrix(original_data, embedding, n_jobs=2), max_K = neighbors + 1)

  coeficients['continuity'] = drcoef['continuity'][neighbors]
  coeficients['trustworthiness'] = drcoef['trustworthiness'][neighbors]
  coeficients['lcmc'] = drcoef['lcmc'][neighbors]
  plot_scatter(result, figsize = (10, 10), title=title, labels = labels, filepath = filepath, filename = filename, coeficients = coeficients)

def create_count_histogram(df : pd.DataFrame, activities : dict, filepath : str, num_subjects : int = 9):
  """
  Creates activity count histograms, for all subjects, and creates a bar plot for each one representing the normalized activity count. Saves in filepath.

    Args:
      df(pd.DataFrame) : the dataFrame that contais all dataset data, already preprocessed.
      activities(dict[int, str]) : a dictionary mapping each activityID value to its activity.
      filepath(str): where to save the plots made.

  """
  values = []
  labels = []
  matrix = []
  for i in range(num_subjects):
    matrix.append([])
  for i in range(0, max(activities.keys()) + 1):
    for j in range(num_subjects):
      matrix[j].append(0)
    values.append(0)
    if i != 0:
      labels.append(activities[i])

  for i in range(df.shape[0]):
    values[int(df["activityID"][i])] += 1
    matrix[int(df["subjectID"][i]) - 1][int(df["activityID"][i])] += 1

  #List with values containing total sum.

  #Matrix does the same, but for every single line

  while 0 in values:
    for i in range(num_subjects):
      matrix[i].pop(values.index(0))
    values.pop(values.index(0))

  while "none" in labels:
    labels.pop(labels.index("none"))

  fig, ax = plt.subplots(1, 1, figsize = (24,14))
  ax.hist(df["activityID"], bins = len(labels))

  # Set title
  ax.set_title("Histograma contagem de atividades")

  # adding labels
  ax.set_xlabel('atividade')
  ax.set_ylabel('contagem')

  # Make some labels.
  rects = ax.patches

  for rect, label, value in zip(rects, labels, values):
    rect.set(height = value)
    height = value
    ax.text(rect.get_x() + rect.get_width()/2, height-0.01, label,
              ha='center', va='bottom')
  # Show plot
  plt.savefig(os.path.join(filepath, "contagem.png"))

  for i in range(num_subjects):
    subject_labels = copy.deepcopy(labels)
    #Normalize and count bins
    for j in range(len(matrix[i])):
      if(matrix[i][j] != 0):
        matrix[i][j] /= values[j]
    #matrix[i] normalized

    while 0 in matrix[i]:
      subject_labels.pop(matrix[i].index(0))
      matrix[i].pop(matrix[i].index(0))

    #matrix[i] w/o zeros e subject_labels as well

    #bar_plot test
    fig = plt.figure(figsize = (24, 14))

    # creating the bar plot
    plt.bar(subject_labels, matrix[i], color ='blue',
            width = 0.4)

    plt.xlabel("Atividade")
    plt.ylabel("Fração da atividade")
    plt.title("Histograma contagem de atividades do sujeito" + str(i + 1))

    # Show plot
    os.makedirs(os.path.join(filepath, "distribuicao"), exist_ok = True)
    plt.savefig(os.path.join(filepath, "distribuicao", "sujeito" + str(i+1) + ".png"))
