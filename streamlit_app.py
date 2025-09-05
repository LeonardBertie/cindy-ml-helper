import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree,export_text
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import json

pages = ["主页","引言：什么是人工智能", "认识鸢尾花数据集", "将你的数据划分为训练集和测试集", "读取数据的完整代码", "模型1:KNN","分类任务的课后习题讨论","模型2:决策树","模型3:支持向量机","模型4:朴素贝叶斯","模型5:多层感知机","集成学习模型"]
page = st.sidebar.radio(
    "选择页面",
    pages
   )

# 页面0：主页
if page == "主页":
    st.title("欢迎来到主页 🎉")
  # 页面1：引言    
elif page == "引言：什么是人工智能":   
    st.title("引言 什么是人工智能")
    st.write("在本学期的第一节课，我们学过————")
    st.image("https://i.postimg.cc/4xwFv5pd/1.png")
    st.image("https://i.postimg.cc/j2xKftDD/image.png")
    st.image("https://i.postimg.cc/7hDgWvky/2.png")
    st.write("这个系统的硬件是计算机平台，软件部分就是程序与数据。我们研究的各种算法就（通过编程语言的形式）被安装在程序中。为了让程序能够运行起来，你就需要提供数据给它。")
    st.write("程序与数据之间主要有两种关系，第一种是“基于规则”形成的关系，典型的如专家系统。在这种系统中，我们要向机器提供数据和推理规则，让机器按照人的思维方式去推理。")
    st.image("https://i.postimg.cc/VvYCFjqf/3.png")
    st.write("专家系统：人类专家将某个领域的知识全部总结出来，按照符号型数据的格式要求存储在知识库中。知识库与推理机形成双向关联。用户对专家系统提出问询时，推理机综合知识库里的知识和用户提出的问题做出推理，并给出正确的答案。")
    st.image("https://i.postimg.cc/CKLJYZk9/4.png")
    st.write("对专家系统而言，不管是知识库里的知识，还是输入给推理机的问题，甚至是推理机给出的结论，都是以符号型数据的形式进行交互与存储的")
    st.image("https://i.postimg.cc/Y9yw62ty/image.png")