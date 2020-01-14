import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


# In[3]:


def create_one_hot_encoding(data):
    encoding = np.zeros((data.shape[0], 4))
    labels = []
    for index, value in enumerate(data):
        labels.append(value[0])
        encoding[index,  int(value[1]) - 1] = 1
    
    return labels, encoding


# In[4]:


def dict_to_matrix(data_dict, labels):
    matrix = []
    for label in labels:
        matrix.append(data_dict[label])
    
    return np.array(matrix).T


# In[5]:


# Data in pickle format
X_train_data = pickle.load(open('cse512hw3/Train_Features.pkl','rb'), encoding='latin')
X_val_data = pickle.load(open('cse512hw3/Val_Features.pkl','rb'), encoding='latin')

# Data in csv format
y_train_data = pd.read_csv('cse512hw3/Train_Labels.csv').to_numpy()
y_val_data = pd.read_csv('cse512hw3/Val_Labels.csv').to_numpy()

# Converting to one hot encoding
y_train_labels, y_train = create_one_hot_encoding(y_train_data)
y_val_labels, y_val = create_one_hot_encoding(y_val_data)

# Unfolding dict to create training, validation dataset
X_train = dict_to_matrix(X_train_data, y_train_labels)
X_val = dict_to_matrix(X_val_data, y_val_labels)


# In[6]:


# Creating numpy array of testing data
X_test_data = pickle.load(open('cse512hw3/Test_Features.pkl','rb'), encoding='latin')

matrix = []
labels = []

for test_label in X_test_data:
    labels.append(test_label)
    matrix.append(X_test_data[test_label])
    
X_test = np.array(matrix).T


# In[7]:


def add_dims(X):
    avg_vec = np.average(X, axis=0)
    max_vec = np.max(X, axis=0)
    min_vec = np.min(X, axis=0)
    median_vec = np.median(X, axis=0)

    return np.vstack((X, avg_vec, max_vec, min_vec, median_vec))


# In[8]:


X_train = add_dims(X_train)
X_val = add_dims(X_val)
X_test = add_dims(X_test)


# In[9]:


X_train.shape


# In[10]:


def create_batch(X, Y, batch_size): 
    """
    X : (d+1) * n
    Y : n * k
    """
    
    dataset = np.hstack((X.T, Y))
    np.random.shuffle(dataset) 
    
    batch_count = dataset.shape[0] // batch_size
    batches = []
    for i in range(batch_count + 1): 
        if i == batch_count and dataset.shape[0] % batch_size != 0:
                batch = dataset[i * batch_size : dataset.shape[0], :]
                
                x_mini_batch = batch[:, :-Y.shape[1]].T
                y_mini_batch = batch[:, -Y.shape[1]:]
                
                batches.append((x_mini_batch, y_mini_batch))
                return batches
                
        
        batch = dataset[i * batch_size : (i + 1) * batch_size, :]
        x_mini_batch = batch[:, :-Y.shape[1]].T
        y_mini_batch = batch[:, -Y.shape[1]:]
        batches.append((x_mini_batch, y_mini_batch))

    return batches

# In[12]:


def calculate_denominator(X, theta):
    den = []
    
    for row in range(X.shape[1]):
        den_i = []
        for theta_ in theta:
            d_i = np.exp(np.matmul(theta_, X[:, row]))
            den_i.append(d_i)
        
        den.append(sum(den_i))
    
    return np.array(den)


# In[13]:


def cost(X, Y, theta):
    n = X.shape[1]
    Y_pred = []
    
    den = 1 + calculate_denominator(X, theta)

    for theta_ in theta:
        y = np.divide(np.exp(np.matmul(theta_, X)), den)
        Y_pred.append(y)
            
    Y_pred = np.array(Y_pred)

        
    y_last = 1 - np.sum(Y_pred, axis=0)
    
    Y_pred = np.vstack((Y_pred, y_last))
    
    loss_curr = (-1 / n) * np.sum(np.log(np.sum(np.multiply(Y_pred.T, Y), axis=1)))
#     print(np.sum(np.log(np.sum(np.multiply(Y_pred.T, Y), axis=1))).shape)
        
    return loss_curr


# In[14]:


def cardinality(X):
    return X.shape[1]


# In[140]:
def calculate_accuracy(x, theta, Y):
    den = 1 + calculate_denominator(x, theta)

    Y_pred = []
    for theta_ in theta:
        y = np.divide(np.exp(np.matmul(theta_, x)), den)
        Y_pred.append(y)

    Y_pred = np.array(Y_pred)
    y_last = 1 - np.sum(Y_pred, axis=0)
    Y_pred = np.vstack((Y_pred, y_last))

    Y_predicted = np.argmax(Y_pred, axis=0)

    return np.sum(Y_predicted == np.argmax(Y, axis=1)) / Y.shape[0]


def logistic_regression(X, Y, batch_size, n0, n1, max_epoch, stopping_criteria):
    """
    X : d * n
    Y : n * k
    """
    d, n = X.shape
    k = Y.shape[1]
    
    # Appending ones to input X
    X = np.divide(X, np.sqrt(np.sum(np.square(X), axis=0)))
    X = np.vstack((X,np.ones((n))))
    
    theta = np.zeros((k - 1, d + 1))

    loss = []
    accuracy = []
    
    for epoch in range(max_epoch):
        eta = n0 / (n1 + epoch)
#         eta = 0.009009009009009009
        batches = create_batch(X, Y, batch_size)
        
        for batch in batches:
            x_batch, y_batch = batch
#             denominator = 1 + np.sum(np.exp(np.dot(theta, x_batch)), axis = 0)  
            denominator = 1 + calculate_denominator(x_batch, theta)
                    
            for i, theta_ in enumerate(theta):
                y_pred = np.divide(np.exp(np.matmul(theta_, x_batch)), denominator)
                
                if cardinality(x_batch):
                    delta = (-1 / cardinality(x_batch)) * np.matmul((y_batch.T[i] - y_pred), x_batch.T)
                    theta[i] = theta_ - eta * delta
        
        loss_curr = cost(X, Y, theta)
        

        print("EPOCH ", epoch , " DONE")
        
        if len(loss) > 1 and loss_curr > (1 - stopping_criteria) * loss[-1]:
            print("CONVERGE")
            break
        
        
        loss.append(loss_curr)
        accuracy.append(calculate_accuracy(X, theta, Y))
        
            
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(len(loss)), loss)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.title('Loss vs Iteration')
    plt.show()
    
    plt.savefig('Validation_Training.png')

    return theta, loss, accuracy



theta_train, loss, accuracy = logistic_regression(X_train, y_train, 16, 0.1, 1, 1000, 0.00001)