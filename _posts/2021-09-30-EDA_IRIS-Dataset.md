---
layout: post
title: "Exploratory Data Analysis of IRIS Dataset"
---




### 1. Dataset:
[iris dataset](https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv("data/iris.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>





```python
df.shape
```




    (150, 5)




```python
df.columns
```




    Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
           'species'],
          dtype='object')




```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sepal_length</th>
      <td>150.0</td>
      <td>5.843333</td>
      <td>0.828066</td>
      <td>4.3</td>
      <td>5.1</td>
      <td>5.80</td>
      <td>6.4</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>150.0</td>
      <td>3.054000</td>
      <td>0.433594</td>
      <td>2.0</td>
      <td>2.8</td>
      <td>3.00</td>
      <td>3.3</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>150.0</td>
      <td>3.758667</td>
      <td>1.764420</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>4.35</td>
      <td>5.1</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>150.0</td>
      <td>1.198667</td>
      <td>0.763161</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>1.30</td>
      <td>1.8</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['species'].unique()
```




    array(['setosa', 'versicolor', 'virginica'], dtype=object)




```python
df['species'].value_counts()
```




    setosa        50
    versicolor    50
    virginica     50
    Name: species, dtype: int64



### 2. 2-D Scatter plot


```python
df.plot(kind='scatter', x='sepal_length', y='sepal_width')
plt.show()
```


    
![png](\img\posts\iris\output_9_0.png)
    



```python
sns.set_style("darkgrid")
sns.FacetGrid(df, hue='species',size=6).map(plt.scatter, "sepal_length", "sepal_width").add_legend()
plt.show()
```


    
![png](\img\posts\iris\output_10_1.png)
    


### 3. 3-D Scatter plot


```python
# pairplot
sns.set_style("dark")
sns.pairplot(df, hue='species', size=3)
plt.show()
```


    
![png](\img\posts\iris\output_12_1.png)
    


Observations:

1. petal_length and petal_width are most clear and useful features to identify various flower types.
2. While setosa can be easily identified(ie; linearly separable), Virgica and vesicolor have some overlap.
3. we can build simple model using if-else to classify the flower types.

### 4. 1-D Scatter plot (Histogram, PDF, CDF)

* Disadvantage of 1-D scatter plot is that it is very hard to make sense as points are overlapping alot.


```python
iris_setosa = df.loc[df["species"] == "setosa"]
iris_virginica = df.loc[df["species"] == "virginica"]
iris_versicolor = df.loc[df["species"] == "versicolor"]

plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa['petal_length']), 'o')
plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')
plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')

plt.show()
```


    
![png](\img\posts\iris\output_15_0.png)
    


**for petal_length**


```python
sns.set_style("dark")
sns.FacetGrid(df, hue="species", size=6).map(sns.distplot, "petal_length").add_legend()
plt.title("Petal Length of Species")
plt.show()

```

    
![png](\img\posts\iris\output_17_1.png)
    


**for petal_width**


```python
sns.set_style("dark")
sns.FacetGrid(df, hue="species", size=6).map(sns.distplot, "petal_width").add_legend()
plt.title("Petal width of Species")
plt.show()

```


    
![png](\img\posts\iris\output_19_1.png)
    


**for sepal_length**


```python
sns.set_style("dark")
sns.FacetGrid(df, hue="species", size=6).map(sns.distplot, "sepal_length").add_legend()
plt.title("Sepal Length of Species")
plt.show()

```

    
![png](\img\posts\iris\output_21_1.png)
    


**for sepal_width**


```python
sns.set_style("dark")
sns.FacetGrid(df, hue="species", size=6).map(sns.distplot, "sepal_width").add_legend()
plt.title("Sepal width of Species")
plt.show()

```

    
![png](\img\posts\iris\output_23_1.png)
    


### Plot CDF of petal_length


```python
counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, density=True)
pdf=counts/(sum(counts))

print("PDF = ",pdf)
print("Bin_edges = ",bin_edges)

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.show()

```

    PDF =  [0.02 0.02 0.04 0.14 0.24 0.28 0.14 0.08 0.   0.04]
    Bin_edges =  [1.   1.09 1.18 1.27 1.36 1.45 1.54 1.63 1.72 1.81 1.9 ]
    


    
![png](\img\posts\iris\output_25_1.png)
    


### Plots of CDF of petal_length for various types of flowers.



```python
# for setosa
counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, density=True)

pdf=counts/(sum(counts))
print("PDF = ",pdf)
print("Bin_edges = ",bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

# for virginica
counts, bin_edges = np.histogram(iris_virginica["petal_length"], bins=10, density=True)

pdf = counts/(sum(counts))
print("PDF = ", pdf)
print("Bins_edges = ", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

# for versicolor
counts, bin_edges = np.histogram(iris_versicolor["petal_length"], bins=10, density=True)

pdf = counts/(sum(counts))
print("PDF = ", pdf)
print("Bins_edges = ", bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)



plt.show()
```

    PDF =  [0.02 0.02 0.04 0.14 0.24 0.28 0.14 0.08 0.   0.04]
    Bin_edges =  [1.   1.09 1.18 1.27 1.36 1.45 1.54 1.63 1.72 1.81 1.9 ]
    PDF =  [0.02 0.1  0.24 0.08 0.18 0.16 0.1  0.04 0.02 0.06]
    Bins_edges =  [4.5  4.74 4.98 5.22 5.46 5.7  5.94 6.18 6.42 6.66 6.9 ]
    PDF =  [0.02 0.04 0.06 0.04 0.16 0.14 0.12 0.2  0.14 0.08]
    Bins_edges =  [3.   3.21 3.42 3.63 3.84 4.05 4.26 4.47 4.68 4.89 5.1 ]
    


    
![png](\img\posts\iris\output_27_1.png)
    


### 5. Mean, Variance and Std-dev


**Compute mean**


```python
print("Means: ")
print(np.mean(iris_setosa["petal_length"]))
print(np.mean(iris_virginica["petal_length"]))
print(np.mean(iris_versicolor["petal_length"]))
# mean with an outlier
print(np.mean(np.append(iris_versicolor["petal_length"],50)))

```

    Means: 
    1.464
    5.552
    4.26
    5.1568627450980395
    

**Compute std-deviation**


```python
print("\nStd-deviation: ")
print(np.std(iris_setosa["petal_length"]))
print(np.std(iris_virginica["petal_length"]))
print(np.std(iris_versicolor["petal_length"]))

```

    
    Std-deviation: 
    0.17176728442867115
    0.5463478745268441
    0.4651881339845204
    

### 6. Median, Percentile, Quantile

**Compute medians**


```python
print("Medians: ")
print(np.median(iris_setosa["petal_length"]))
print(np.median(iris_virginica["petal_length"]))
print(np.median(iris_versicolor["petal_length"]))
# mean with an outlier
print(np.median(np.append(iris_versicolor["petal_length"],50)))

```

    Medians: 
    1.5
    5.55
    4.35
    4.4
    

**Compute Quantiles**


```python
print("Quantiles: ")
print(np.percentile(iris_setosa["petal_length"], np.arange(0,100,25)))
print(np.percentile(iris_virginica["petal_length"], np.arange(0,100,25)))
print(np.percentile(iris_versicolor["petal_length"], np.arange(0,100,25)))
```

    Quantiles: 
    [1.    1.4   1.5   1.575]
    [4.5   5.1   5.55  5.875]
    [3.   4.   4.35 4.6 ]
    


```python
print("\n90th Percentiles:")
print(np.percentile(iris_setosa["petal_length"],90))
print(np.percentile(iris_virginica["petal_length"],90))
print(np.percentile(iris_versicolor["petal_length"], 90))
```

    
    90th Percentiles:
    1.7
    6.31
    4.8
    

**Compute Absolute Deviation**


```python
from statsmodels import robust
print("Median Absolute Deviation")
print(robust.mad(iris_setosa["petal_length"]))
print(robust.mad(iris_virginica["petal_length"]))
print(robust.mad(iris_versicolor["petal_length"]))

```

    Median Absolute Deviation
    0.14826022185056031
    0.6671709983275211
    0.5189107764769602
    

**Box Plot**


```python
sns.boxplot(x="species", y="petal_length", data=df)
plt.show()
```


    
![png](\img\posts\iris\output_42_0.png)
    


**Violin plot**


```python
sns.violinplot(x="species",y='petal_length', data=df)
plt.show()
```


    
![png](\img\posts\iris\output_44_0.png)
    



```python

```
