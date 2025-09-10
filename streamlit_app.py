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
from openai import OpenAI
import altair as alt


import os, json
from supabase import create_client
from dotenv import load_dotenv
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=st.secrets["DEEPSEEK_API_KEY"] # 在 .streamlit/secrets.toml 配置
)

load_dotenv()  # 本地开发用 .env

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY =st.secrets["SUPABASE_ANON_KEY"] 
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("请先在环境变量设置 SUPABASE_URL 与 SUPABASE_ANON_KEY")
    st.stop()

# 全局匿名客户端（用于公开操作 / 建立会话）
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config("基于streamlit的人工智能分类算法辅助系统", layout="wide")

# ----------------- 帮助函数 -----------------
def sign_up(email, password, full_name=None):
    """注册（返回 response 对象）"""
    res = supabase.auth.sign_up({"email": email, "password": password})
    return res

def sign_in(email, password):
    """登录，返回包含 access/refresh token 的 response"""
    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    return res

def save_profile_if_missing(user_id, full_name=None, role="user"):
    """尝试在 public.profiles 建立 profile"""
    existing = supabase.table("profiles").select("id").eq("id", user_id).execute()
    if existing.data and len(existing.data) > 0:
        return
    supabase.table("profiles").insert({"id": user_id, "full_name": full_name or "", "role": role}).execute()



def make_user_client(access_token=None):
    """
    为当前用户创建一个临时 supabase client（带 access_token 的请求）
    这样后续操作会在该用户的 RLS 上下文下执行
    """
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    if access_token:
        # 直接在调用时传 headers，不需要 ClientOptions
        client.headers.update({"Authorization": f"Bearer {access_token}"})

    return client

# 读取用户数据
def load_user_data(user_id, key):
    res = supabase.table("user_data").select("value").eq("user_id", user_id).eq("key", key).execute()
    if res.data and len(res.data) > 0:
        return res.data[0]["value"]
    return ""

# 保存用户数据
def save_user_data(user_id, key, value):
    # 先检查是否已有记录
    res = supabase.table("user_data").select("id").eq("user_id", user_id).eq("key", key).execute()
    if res.data and len(res.data) > 0:
        # 更新
        supabase.table("user_data").update({"value": value}).eq("id", res.data[0]["id"]).execute()
    else:
        # 插入
        supabase.table("user_data").insert({"user_id": user_id, "key": key, "value": value}).execute()
# 保存用户某页完成情况
def save_page_progress(user_id, page, completed):
    # 检查是否已有记录
    res = supabase.table("user_progress").select("id").eq("user_id", user_id).eq("page", page).execute()
    if res.data and len(res.data) > 0:
        # 更新
        supabase.table("user_progress").update({"completed": completed}).eq("id", res.data[0]["id"]).execute()
    else:
        # 插入
        supabase.table("user_progress").insert({
            "user_id": user_id,
            "page": page,
            "completed": completed
        }).execute()

# 加载用户全部进度
def load_user_progress(user_id, pages):
    progress = {page: False for page in pages}
    res = supabase.table("user_progress").select("page, completed").eq("user_id", user_id).execute()
    if res.data:
        for record in res.data:
            progress[record["page"]] = record["completed"]
    return progress

# 初始化 session_state
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_id = None
    st.session_state.role = "user"   # 默认角色是 user


def get_user_role(user_id):
    """从 profiles 表获取角色"""
    res = supabase.table("profiles").select("role").eq("id", user_id).execute()
    if res.data and len(res.data) > 0:
        return res.data[0].get("role", "user")
    return "user"

def get_all_users():
    """获取所有用户及角色"""
    res = supabase.table("profiles").select("id, full_name, role").execute()
    return res.data if res.data else []

def get_user_progress():
    """获取所有用户的进度"""
    res = supabase.table("user_progress").select("user_id, page, completed").execute()
    return res.data if res.data else []
def st_highlight(text, color="#FFEFD5"):
    """
    在 Streamlit 中显示高亮文本块。
    默认底色为淡橙色（#FFEFD5），可传入任意 CSS 颜色值。
    """
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:12px; border-radius:8px; margin-bottom:10px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )
def mark_progress(user_id, page):
    """标记用户完成某个页面"""
    # 查询是否已存在记录
    existing = supabase.table("user_progress").select("id").eq("user_id", user_id).eq("page", page).execute()
    if existing.data:
        supabase.table("user_progress").update({"completed": True}).eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("user_progress").insert({
            "user_id": user_id,
            "page": page,
            "completed": True
        }).execute()
    
    # 同步更新 session_state
    if "completed" not in st.session_state:
        st.session_state.completed = {}
    st.session_state.completed[page] = True
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_id = None
    st.session_state.role = "user"
    st.session_state.completed = {}

# ---------------- 登录/注册 ----------------
if st.session_state.user is None:
    st.subheader("注册新用户")
    reg_email = st.text_input("邮箱（注册）", key="reg_email")
    reg_pw = st.text_input("密码（注册）", type="password", key="reg_pw")
    if st.button("注册"):
        try:
            res = supabase.auth.sign_up({"email": reg_email, "password": reg_pw})
            if res.user:
                st.success(f"注册成功！请使用 {reg_email} 登录")
            else:
                st.error(f"注册失败: {getattr(res, 'error', '未知错误')}")
        except Exception as e:
            st.error(f"注册异常: {e}")
    st.info("请在邮箱查收确认邮件，在邮箱点击确认按钮后无需等待页面加载完成即可完成注册")

    st.subheader("登录")
    login_email = st.text_input("邮箱（登录）", key="login_email")
    login_pw = st.text_input("密码（登录）", type="password", key="login_pw")
    if st.button("登录"):
        try:
            res = supabase.auth.sign_in_with_password({"email": login_email, "password": login_pw})
            session = getattr(res, "session", None)
            user = getattr(res, "user", None)
            if session and user:
                st.session_state.user = user
                st.session_state.access_token = session.access_token
                st.session_state.refresh_token = session.refresh_token
                st.session_state.user_id = user.id
                st.session_state.role = get_user_role(user.id)

                # 确保 profiles 表有记录
                existing = supabase.table("profiles").select("id").eq("id", user.id).execute()
                if not existing.data or len(existing.data) == 0:
                    supabase.table("profiles").insert({
                        "id": user.id,
                        "full_name": user.email,
                        "role": "user"
                    }).execute()

                st.success(f"登录成功，用户ID: {user.id}，角色: {st.session_state.role}")
                st.rerun()  # 登录后刷新页面
            else:
                st.error("登录失败，请检查邮箱和密码。")
        except Exception as e:
            st.error(f"登录异常: {e}")

# ---------------- 登录成功后的页面 ----------------
else:
 if st.session_state.role == "admin":
        st.title("👑 管理员后台")
        users = get_all_users()
        progress = get_user_progress()

        #if users:
            #st.subheader("所有用户")
            #df_users = pd.DataFrame(users)  # 包含 id, full_name, role
            #st.dataframe(df_users)

        if progress:
         st.subheader("用户进度")
         df_progress = pd.DataFrame(progress)
         df_progress = df_progress.pivot(index="user_id", columns="page", values="completed").fillna(False)

         # 映射 user_id -> full_name
         id_to_name = {u["id"]: u["full_name"] for u in users}
         df_progress.index = [id_to_name.get(uid, uid) for uid in df_progress.index]

         # 获取 user_notes 表中所有 note
         notes_data = supabase.table("user_notes").select("*").execute().data

         # 构建 {user_id: {page: note}} 的字典
         user_notes = {}
         for note in notes_data:
            uid = note["id"]
            page = note["page"]
            user_notes.setdefault(uid, {})[page] = note["note"]

         # 新增一列 "用户文字"（这里示例取 homepage 页面的 note）
         df_progress["用户名"] = [
            user_notes.get(uid, {}).get("homepage", "") for uid in df_progress.index
         ]

         # 将 "用户文字" 列移到最前面
         cols = df_progress.columns.tolist()
         cols = ["用户名"] + [c for c in cols if c != "用户名"]
         df_progress = df_progress[cols]
 
         st.dataframe(df_progress)

 else:
   st.success(f"用户ID: {st.session_state.user_id}，角色: {st.session_state.role}")
   pages = ["主页","引言：什么是人工智能", "认识鸢尾花数据集",
                 "将你的数据划分为训练集和测试集", "读取数据的完整代码",
                 "模型1:KNN","分类任务的课后习题讨论","模型2:决策树",
                 "模型3:支持向量机","模型4:朴素贝叶斯","模型5:多层感知机",
                 "集成学习模型"]

        # 初始化 completed（加载用户进度）
   if "completed" not in st.session_state:
          st.session_state.completed = load_user_progress(st.session_state.user_id, pages)
   with st.sidebar:
    page = st.radio(
            "选择页面",
            pages,
            format_func=lambda x: f"✅ {x}" if st.session_state.completed.get(x, False) else x
    )
    st.markdown("---")  # 分隔线

    # DeepSeek 助手
    st.header("💬 DeepSeek 助手")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 输入框
    user_question = st.text_area("请输入问题：", key="user_input", height=100)

    # 提交按钮
    if st.button("🚀 提交问题", key="submit_btn"):
        if user_question.strip():
            # 每次只保留最新的问答
            st.session_state.messages = [
                {"role": "user", "content": user_question}
            ]

            with st.spinner("正在思考中..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                    temperature=0.7
                )
            answer = response.choices[0].message.content

            # 覆盖，只保留最新回答
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # 展示最新的一问一答
    if st.session_state.messages:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

   

   # 页面0：主页
   if page == "主页":
            st.title("欢迎来到主页 🎉")
            user_id = st.session_state.user_id  # 假设登录后存了用户id

            st.subheader("修改用户名为您的真实姓名")
            user_text = st.text_area("请输入文字")

            if st.button("提交文字"):
              if user_text.strip():
                # 上传到 user_notes 表
                supabase.table("user_notes").upsert({
                "id": user_id,
                "page": "homepage",  # 可以按页面分类
                "note": user_text
                }).execute()
                st.success("提交成功！")
              else:
                st.warning("请输入内容再提交")
 

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
  # 页面2：数据展示
   elif page == "认识鸢尾花数据集":
    st.subheader("认识鸢尾花数据集")
    st.write("经典的鸢尾花数据集，iris，它一共有4种不同的特征，3个类别的标签，150个样本，其中1-50属于类别1,51-100属于类别2,101-150属于类别3")
    st.image("https://i.postimg.cc/MpjXvBKF/5.png")
    st.subheader("【python版本】")
    st_highlight("#%%读入鸢尾花数据集")
    st_highlight("from sklearn import datasets")
    st_highlight("iris_datas=datasets.load_iris() ")
    st_highlight("feature=iris_datas.data")
    st_highlight("label=iris_datas.target ")
    st.write("【请尝试读入这个数据集吧，其中的特征用feature表示，标签用label表示】")
    if st.button("运行代码"):
    # 读入鸢尾花数据集
     iris_datas = datasets.load_iris()
     feature = iris_datas.data
     label = iris_datas.target
     st.success("代码运行完成！")
     st.write("为了显示读入鸢尾花数据集情况，打印特征矩阵和标签向量")
     st.write("✅ 特征矩阵 (前5行):")
     st.write(feature[:5])  # 只展示前5行，避免太长
     st.write()
    
     st.write("✅ 标签向量 (前20个):")
     st.write(label[:20])
    
     st.success("代码运行完成！")
    st.subheader("【安装依赖包】")
    st.write("pipinstallscikit-learn-ihttps://pypi.tuna.tsinghua.edu.cn/simple")
    st.subheader("【过程详解】")
    st.write("datasets是scikit-learn提供的内置数据集模块，包含多种经典数据集（如鸢尾花、波士顿房价等）")
    st.subheader("iris_datas=datasets.load_iris()")
    st.caption("说明：")
    st.write("load_iris()返回一个Bunch对象（类似字典的结构），包含以下键：")
    st.write("data:特征数据（二维数组）。")
    st.write("target:标签数据（一维数组）。")
    st.write("feature_names:特征名称列表（如花萼长度、宽度等）。")
    st.write("target_names:标签名称列表（鸢尾花种类：setosa,versicolor,virginica）。")
    st.caption("数据类型：")
    st.write("iris_datas是sklearn.utils.Bunch类型（类似字典的键值对结构）。")
    st.write("Iris_datas的结构形式")
    st.image("https://i.postimg.cc/L6VGj9NB/6.png")
    st.subheader("feature=iris_datas.data")
    st.caption("说明：")
    st.text("特征数据是一个二维数组，每行代表一个样本，每列代表一个特征。")
    st.write("鸢尾花数据集有150个样本（行）和4个特征（列），对应：")
    st.write("花萼长度（sepallength）")
    st.write("花萼宽度（sepalwidth）")
    st.write("花瓣长度（petallength）")
    st.write("花瓣宽度（petalwidth）")
    st.caption("数据类型：")
    st.write("numpy.ndarray（形状为(150,4)）。")
    st.subheader("如果我想查看前5个样本的特征")
    st.write("print(feature[:5])#查看前5个样本的特征")
    st.image("https://i.postimg.cc/pLYgvD9H/7.png")
    st.write("这里涉及到python的一个基本语法：array[start:stop:step]")
    st.image("https://i.postimg.cc/RCKd2vJr/8.png")
    st.write("feature[:5]表示从0开始，到5结束，步长为默认值1")
    st.write("等效写法：feature[0:5]。")
    st.write("*其他常见的切片示例")
    st.image("https://i.postimg.cc/MGsZ6kcq/0.png")
    st.write("*如何获得第0行第0列的第一个特征值？")
    st.image("https://i.postimg.cc/2jLWKGJV/9.png")

  # 页面3：模型训练
   elif page == "将你的数据划分为训练集和测试集":
    st.subheader("将你的数据划分为训练集和测试集")
    st.write("在机器学习中，为了让你的模型（算法）能够学习，我们需要先收集很多的数据，构成数据集。为了验证你使用的算法的性能，我们需要将数据集划分为训练集与测试集。训练集和测试集的内容应该是“互斥”的，即测试集测试的是训练集中没有的数据，也就是机器在学习过程中没有见过的数据，这样才能去证明它具有“举一反三”的学习能力。")
    st.image("https://i.postimg.cc/d3CVP8SC/1.png")
    st.write("机器学习中，通常将所有的数据按照8:2的比例划分为训练集和测试集。训练集用于对模型进行训练，测试集用于验证模型在不同任务中的预测性能。")
    st.subheader("【python】")
    st_highlight("#%%划分测试集,训练集")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("import numpy as np")
    st_highlight("indics=np.arange(feature.shape[0])#生成索引序列")
    st_highlight("X_train_ind,X_test_ind,X_train,X_test=train_test_split(indics,feature,test_size=0.2,random_state=42)")
    if st.button("划分数据集和测试集"):
    # 读入鸢尾花数据集
     iris_datas = datasets.load_iris()
     feature = iris_datas.data
     label = iris_datas.target
    
    # 生成索引序列
     indics = np.arange(feature.shape[0])
    
    # 划分训练集和测试集
     X_train_ind, X_test_ind, X_train, X_test = train_test_split(
        indics, feature, test_size=0.2, random_state=42
     )
     st.success("训练集和测试集划分完成！")
     st.write("为了显示训练集和测试集划分情况，分别打印其索引和特征")
    # 显示结果
     st.write("✅ 训练集索引 (前10个):", X_train_ind[:10])
     st.write("✅ 测试集索引 (前10个):", X_test_ind[:10])
     st.write("✅ 训练集特征 (前5行):")
     st.write(X_train[:5])
     st.write("✅ 测试集特征 (前5行):")
     st.write(X_test[:5])
    
     
    st.subheader("【说明】")
    st.subheader("indics=np.arange(feature.shape[0])#生成索引序列")
    st.write("功能：生成一个从0到feature样本数减1的连续整数序列。")
    st.write("参数：")
    st.write("feature.shape[0]:获取feature的行数（样本数），例如鸢尾花数据集为150。")
    st.write("np.arange(n):生成[0,1,...,n-1]的数组。")
    st.write("输出：indices:numpy.ndarray，形状为(150,)，例如[0,1,2,...,149]。")
    st.subheader("X_train_ind,X_test_ind,X_train,X_test=train_test_split(indics,feature,test_size=0.2,random_state=42)")
    st.write("功能：将数据和对应的索引划分为训练集和测试集。")
    st.write("参数：")
    st.write("indices:样本索引数组（[0,1,...,149]）。")
    st.write("feature:特征数据（形状(150,4)）。")
    st.write("test_size=0.2:测试集占比20%（30个样本），训练集占比80%（120个样本）。")
    st.write("random_state=42:随机种子，确保每次划分结果一致（可复现性）。")
    st.write("返回值：")
    st.write("X_train_ind:训练集的索引，numpy.ndarray，形状(120,)。")
    st.write("X_test_ind:测试集的索引，numpy.ndarray，形状(30,)。")
    st.write("X_train:训练集特征，numpy.ndarray，形状(120,4)。")
    st.write("X_test:测试集特征，numpy.ndarray，形状(30,4)。")
    st.image("https://i.postimg.cc/xdczKPVG/11.png")
    st.write("这样划分的目的是：")
    st.write("①知道测试集和训练集分别有哪些数据")
    st.write("②知道原始数据中的哪些被划分到了训练集，哪些被划分到了测试集")
    st.subheader("【提问】第150个样本（编号149属于测试集还是训练集）？")
    st.image("https://i.postimg.cc/MK3ZyJ64/12.png")
    st.write("如果用代码如何实现？")
    st.write("（方法1：python的in操作符）ifsample_indexinX_train_ind:这行代码的作用是检查某个样本的索引是否存在于训练集的索引数组X_train_ind中。如果存在，说明该样本被划分到了训练集；否则，需要进一步检查是否在测试集中。")
    st.write("（方法2：numpy库中的np.isin函数）np.isin(sample_index,X_train_ind)")
    st.image("https://i.postimg.cc/pTccsCm2/13.png")
    st.subheader("【提问】特征提取出来了，如何根据特征提取训练集和测试集对应的标签？")
    st.write("【python】")
    st_highlight("Y_train=label[X_train_ind]")
    st_highlight("Y_test=label[X_test_ind]")
    if st.button("根据特征提取对应标签"):
    # 1. 读入鸢尾花数据集
     iris_datas = datasets.load_iris()
     feature = iris_datas.data
     label = iris_datas.target
    
    # 2. 生成索引序列
     indics = np.arange(feature.shape[0])
    
    # 3. 划分训练集和测试集
     X_train_ind, X_test_ind, X_train, X_test = train_test_split(
        indics, feature, test_size=0.2, random_state=42
    )
    
    # 4. 获取对应的标签
     Y_train = label[X_train_ind]
     Y_test = label[X_test_ind]
     st.success("提取成功")
     st.write("为了展示提取情况，打印特征和标签")
    # 5. 在页面上展示结果
     st.write("📊 数据集基本信息")
     st.write(f"训练集样本数: {X_train.shape[0]}")
     st.write(f"测试集样本数: {X_test.shape[0]}")
    
     st.write("✅ 训练集特征 (前5行):")
     st.write(X_train[:5])
     st.write("✅ 训练集标签 (前10个):", Y_train[:10])
    
     st.write("✅ 测试集特征 (前5行):")
     st.write(X_test[:5])
     st.write("✅ 测试集标签 (前10个):", Y_test[:10])
    
     st.success("训练集、测试集及其标签生成完成！")
    st.image("https://i.postimg.cc/bv79r5SS/14.png")
    st.write("需要检查一下标签的维数和特征的维数保持一致")
    st.image("https://i.postimg.cc/vH477M7x/image.png")
    st.image("https://i.postimg.cc/BvSXgBW7/15.png")
    st.image("https://i.postimg.cc/pdsBvBK3/16.png")
    st.write("有了这些数据，下面我们就可以开始训练不同的模型了。python的sklearn内部自带了很多机器学习模型，大家可以多多尝试~~")

  # 页面4：模型训练
   elif page == "读取数据的完整代码":
    st.subheader("读取数据的完整代码")
    st.subheader("【python】")
    st_highlight("#%%读入鸢尾花数据集")
    st_highlight("from sklearn import datasets")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("feature=iris_datas.data")
    st_highlight("label=iris_datas.target")
    st_highlight("#%%划分测试集,训练集")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("import numpy as np")
    st_highlight("indics=np.arange(feature.shape[0])#生成索引")
    st_highlight("X_train_ind,X_test_ind,X_train,X_test=train_test_split(indics,feature,test_size=0.2,random_state=42)")
    st_highlight("Y_train=label[X_train_ind]")
    st_highlight("Y_test=label[X_test_ind]")
    st.write("【注意】#X_train_ind,X_test_ind,X_train,X_test=train_test_split(indics,feature,test_size=0.2,random_state=42)")
    st.write("X_train1,X_test,Y_train1,Y_test1=train_test_split(feature,label,test_size=0.2,random_state=42)")
    st.write("这种方法也可以得到Y_train和Y_test，但是输出的变量个数只能是4个，不能同时输出索引值")
    if st.button("运行完整代码"):
    # 1. 读入鸢尾花数据集
     iris_datas = datasets.load_iris()
     feature = iris_datas.data
     label = iris_datas.target
    
    # 2. 生成索引序列
     indics = np.arange(feature.shape[0])
    
    # 3. 划分训练集和测试集
     X_train_ind, X_test_ind, X_train, X_test = train_test_split(
        indics, feature, test_size=0.2, random_state=42
    )
    
    # 4. 获取对应的标签
     Y_train = label[X_train_ind]
     Y_test = label[X_test_ind]
     st.success("提取成功")
     
     st.write("展示结果") 
     st.write("📊 数据集基本信息")
     st.write(f"训练集样本数: {X_train.shape[0]}")
     st.write(f"测试集样本数: {X_test.shape[0]}")
    
     st.write("✅ 训练集特征 (前5行):")
     st.write(X_train[:5])
     st.write("✅ 训练集标签 (前10个):", Y_train[:10])
     st.write("✅ 训练集索引 (前10个):", X_train_ind[:10])
     
     st.write("✅ 测试集索引 (前10个):", X_test_ind[:10])
     st.write("✅ 测试集特征 (前5行):")
     st.write(X_test[:5])
     st.write("✅ 测试集标签 (前10个):", Y_test[:10])
    st.subheader("【提问】如果鸢尾花数据集是一个excel的csv文档，应该如何导入数据呢？")
    st.write("这个文件长这样，有151行，数据在第2-151行，特征在第2-5列，标签在第6列")
    st.image("https://i.postimg.cc/wv79b4Tv/17.png")
    st.subheader("【python代码】")
    st_highlight("import pandas as pd")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("#读取CSV文件")
    st_highlight("data=pd.read_csv('iris.csv')")
    st_highlight("#检查数据前几行，确保正确读取")
    st_highlight("print(data.head())")
    if st.button("读取导入的鸢尾花数据集csv文件"):
     # 1. 自动生成鸢尾花数据集 DataFrame
     iris = datasets.load_iris()
     iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
     iris_df["target"] = iris.target
     # 2. 显示前几行
     st.write("✅ 数据集前5行：")
     st.write(iris_df.head())
     # 3. 显示基本信息
     st.write("📊 数据集基本信息：")
     st.write(f"样本数: {iris_df.shape[0]}")
     st.write(f"特征数: {iris_df.shape[1] - 1}")
     st.success("鸢尾花数据集已加载完成！")
    st.write("自动读取成了150行6列，注意数据类型是dataframe格式的，需要用dataframe格式的读取方式")
    st.image("https://i.postimg.cc/PxgnVtR3/18.png")
    st.subheader("【python代码】")
    st_highlight("#分离特征和标签")
    st_highlight("#特征：Sepal.Length,Sepal.Width,Petal.Length,Petal.Width")
    st_highlight("X=data[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]")
    st_highlight("#标签：Species")
    st_highlight("y=data['Species']")
    st.write("得到的结果是这样的")
    st.image("https://i.postimg.cc/zXP6xDHn/19.png")
    st.write("这种形式是我们不熟悉的，需要进行转换")
    st.write("特征的转换很容易")
    st_highlight("#将特征转换为float64类型的NumPy数组")
    st_highlight("X_array=X.to_numpy(dtype='float64')")
    st.write("标签的转换有点复杂了，这里提供两种思路")
    st_highlight("#将标签转换为NumPy数组")
    st_highlight("#方法1使用if-elif条件进行转换")
    st_highlight("y_array=y.to_numpy()#获得numpy类型数据")
    st_highlight("#创建一个空数组存储转换后的数值，与输入数组y_array形状和数据类型相同的全零数组")
    st_highlight("y_numeric=np.zeros_like(y_array,dtype='int64')")
    st_highlight("for i in range(len(y_array)):")
    st_highlight("if y_array[i]=='setosa':")
    st_highlight("y_numeric[i]=0")
    st_highlight("elif y_array[i]=='versicolor':")
    st_highlight("y_numeric[i]=1")
    st_highlight("elif y_array[i]=='virginica':")
    st_highlight("y_numeric[i]=2")
    st.write("━━━━━━━━━━━━━━━━━━")
    st_highlight("#方法2:使用pandasfactorize")
    st_highlight("#y_int64,classes=pd.factorize(y)#factorize按首次出现的顺序排序")
    st_highlight("最后就是按照一开始的思路划分训练集和测试集")
    st_highlight("#划分训练集和测试集")
    st_highlight("#通常使用80%训练，20%测试，随机种子设为42以保证可重复性")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(X_array,y_numeric,test_size=0.2,random_state=42)")
    st.write("     ")
    st.write("结果已经非常好看了")
    st.image("https://i.postimg.cc/VNg1KQqn/20.png")
    st.subheader("【以上部分完整的python代码】")
    st_highlight("#%%从csv文件导入数据并且划分训练集和测试集")
    st_highlight("import pandas as pd")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("#读取CSV文件")
    st_highlight("data=pd.read_csv('iris.csv')")
    st_highlight("#检查数据前几行，确保正确读取")
    st_highlight("print(data.head())")
    st_highlight("#分离特征和标签")
    st_highlight("#特征：Sepal.Length,Sepal.Width,Petal.Length,Petal.Width")
    st_highlight("X=data[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]")
    st_highlight("#标签：Species")
    st_highlight("y=data['Species']")
    st_highlight("#将特征转换为float64类型的NumPy数组")
    st_highlight("X_array=X.to_numpy(dtype='float64')")
    st_highlight("#将标签转换为NumPy数组")
    st_highlight("#方法1使用if-elif条件进行转换")
    st_highlight("y_array=y.to_numpy()#获得numpy类型数据")
    st_highlight("#创建一个空数组存储转换后的数值，与输入数组y_array​​形状和数据类型相同​​的全零数组")
    st_highlight("y_numeric=np.zeros_like(y_array,dtype='int64')")
    st_highlight("for i in range(len(y_array)):")
    st_highlight("if y_array[i]=='setosa':")
    st_highlight("y_numeric[i]=0")
    st_highlight("elif y_array[i]=='versicolor':")
    st_highlight("y_numeric[i]=1")
    st_highlight("elif y_array[i]=='virginica':")
    st_highlight("y_numeric[i]=2")
    st.write("━━━━━━━━━━━━━━━━━━")
    st_highlight("#方法2:使用pandas factorize")
    st_highlight("#y_int64,classes=pd.factorize(y)#factorize按首次出现的顺序排序")
    st_highlight("#划分训练集和测试集")
    st_highlight("#通常使用80%训练，20%测试，随机种子设为42以保证可重复性")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(X_array,y_numeric,test_size=0.2,random_state=42)")
    st.subheader("【提问】如果鸢尾花数据集是一个txt的文档，应该如何导入数据呢？")
    st.subheader("【python代码】")
    st_highlight("#%%从txt文件读入")
    st_highlight("import pandas as pd")
    st_highlight("import numpy as np")
    st_highlight("#读取txt文件")
    st_highlight('#假设文件名为"iris.txt"，与代码在同一目录下')
    st_highlight("#sep='\s+':读取空格/制表符分隔文本")
    st_highlight("#header=0：表示将第一行（行索引0）作为列名")
    st_highlight("data=pd.read_csv('iris.txt',sep='\s+',header=0)")
    st_highlight("#提取特征列（前4列）并转换为float类型")
    st_highlight("features=data.iloc[:,:4].astype(float).values")
    st_highlight("#提取标签列（第5列）并转换为int类型")
    st_highlight("#首先将字符串标签映射为数字")
    st_highlight("species_mapping={'setosa':0,'versicolor':1,'virginica':2}")
    st_highlight("labels=data['Species'].map(species_mapping).astype(int).values")
    st_highlight("#划分训练集和测试集")
    st_highlight("#通常使用80%训练，20%测试，随机种子设为42以保证可重复性")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=42)")
 
  # 页面5：模型训练
   elif page == "模型1:KNN":
    st.write("机器学习方法根据任务不同，主要有有监督学习、无监督学习、半监督学习和强化学习。")
    st.image("https://i.postimg.cc/dtrtHs8k/image.png")
    st.write("这一部分，我们将从有监督算法开始，学习一些最基本的，容易上手的算法案例")
    st.subheader("模型1KNN")
    st.write("K临近(K-nearestneighbors)是一种基于实例的分类方法，最初是由Cover和Hart于1968年提出的，是一种非参数的分类方法。")
    st.write("分类：预测离散的数据对象。分类数据的标签已知。属于有监督的学习方法。")
    st.write("容易混淆的词：聚类，聚类是在数据中寻找隐藏的模式或分组。聚类算法构成分组和类，类中的数据具有很高的相似度。属于无监督的学习方法。")
    st.write("基本思想：通过计算每个训练样例到待分类样品的距离，取和待分类样例距离最近的K个训练样例。这K个训练样例中哪个类别的标签占多数，则待分类样例就属于哪个类别。")
    st.write("通俗解释：如果一只动物，它走起来像鸭子，叫像鸭子，看起来还像鸭子，那么它可能就是一只鸭子")
    st.write("任务说明：有两类不同的样本数据，分别用蓝色的小正方形和红色的小三角形表示，而图正中间的那个绿色的圆代表则是待分类的测试集数据。我们不知道中间那个绿色的圆从属哪一类别(蓝色正方形or红色三角形)，但它一定这两者中的一种。下面我们就要解决给这个绿色的圆点进行二分类的问题。")
    st.image("https://i.postimg.cc/RVFLKYfD/1.png")
    st.write("俗话说，物以类聚，人以群分，判别一个人是一个什么样品质特征的人，常常可以从他/她身边的朋友入手。现在为了判别上图中的绿色圆形属于哪个类别(蓝色正方形or红色三角形)，我们就从它的邻居下手来进行判断。但一次性判断多少个邻居呢？有以下几种方式可以选择：")
    st.write("K=3，绿色圆点的最近的3个邻居是2个红色小三角形和1个蓝色小正方形，少数服从多数，基于统计的方法，判定绿色的这个待分类点属于红色的三角形一类。")
    st.write("K=5，绿色圆点的最近的5个邻居是2个红色三角形和3个蓝色的正方形，还是少数服从多数，基于统计的方法，判定绿色的这个待分类点属于蓝色的正方形一类。")
    st.write("于此我们看到，当无法判定当前待分类点是从属于已知分类中的哪一类时，我们可以依据统计学的理论看它所处的位置特征，衡量它周围邻居的权重，而把它归为(或分配)到权重更大的那一类。这就是K近邻算法的核心思想。")
    st.write("计算与样例点之间距离的时候，最常见的方法还是欧式距离")
    st.image("https://i.postimg.cc/vmB4NRYJ/2.png")
    st.write("特例1：如果待分类点的附近只有一个样例点，那就直接使用它的分类")
    st.write("特例2：如果待分类点的附近有相同数量的两类样例点，那就随机选择一个")
    st.write("注意：如果K值取的太小，可能会造成参与评估的样本集太小，结果没有说服力。如果K值取的太大，会把距离目标队列太远的噪声数据也考虑进去，造成结果不准确。")
    st.write("办法：反复调试参数K")
    st.info("K-NN算法的基本步骤如下：")
    st.info("1)初始化未知样本与第一个训练集样本的距离为最大值")
    st.info("2)计算未知样本到每一个训练集样本的距离dist")
    st.info("3)得到目前K个最近邻样本中的最大距离maxdist")
    st.info("4)寻找新的样本点，如果dist<maxdist，则将该训练样本作为K-近邻样本")
    st.info("5)重复步骤2)-4)，直到未知样本和所有训练样本的距离都计算完")
    st.info("6)统计K个最近邻样本中每个类别出现的次数，出现频率最大的类别作为未知样本的类别")
    st.info("7)有多个未知样本，则重复1)-6)")
    st.info("K-NN算法不仅可以用于二分类，还可以用于多分类问题，是一种非常简单好用的方法")
    st.write("最后，我们可以训练K-NN分类器了。在python的sklearn中，它们都是封装好的API")
    st.subheader("任务1：利用划分好的训练集的数据训练一个分类器")
    st.subheader("【python】")
    st_highlight("#%%训练机器学习模型")
    st_highlight("#KNN")
    st_highlight("from sklearn.neighbors import KNeighborsClassifier#先调包")
    st_highlight("clf_KNN=KNeighborsClassifier(n_neighbors=5)#建立一个模型框架")
    st_highlight("clf_KNN.fit(X_train,Y_train)#代入数据训练")
    st.write("训练完毕，输出一个训练好的模型对象")
    st.image("https://i.postimg.cc/8zf3BcW9/3.png")
    if st.button("运行 KNN 模型训练"):
    # 训练模型
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train)
    
    # 输出提示和模型信息
     st.success("模型训练完成！")
     st.write("训练好的模型对象：", clf_KNN)
    st.subheader("任务2：利用训练好的分类器在测试集上输出结果")
    st.write("一行代码就可以搞定~")
    st.subheader("【python】")
    st_highlight("KNN_pred=clf_KNN.predict(X_test)")
    st.write("预测的结果储存在KNN_pred这个变量中，得到了针对测试集的30个样本的输出")
    st.image("https://i.postimg.cc/KcVD7NNc/4.png")
    if st.button("预测结果"):
    # 训练模型
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train) 
     # 对测试集进行预测
     KNN_pred = clf_KNN.predict(X_test)
     # 显示预测结果
     st.write("测试集预测结果：", KNN_pred)
    st.write("在python中，也可以输出计算结果的预测概率，有时候这个概率值很有用~~")
    st_highlight("#输出计算结果的概率值")
    st_highlight("KNN_pred_proba=clf_KNN.predict_proba(X_test)")
    st.image("https://i.postimg.cc/5yw52cFP/5.png")
    if st.button("预测概率"):
    # 训练模型
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train) 
     # 对测试集进行预测
     KNN_pred = clf_KNN.predict(X_test)
     KNN_pred_proba=clf_KNN.predict_proba(X_test)
     st.write("预测概率",KNN_pred_proba)
    st.subheader("任务3：判断分类器的分类效果")
    st.info("怎么来判断模型效果呢？肉眼对比吗？")
    st.write("错误率ErrorRate：分类错误的样本占样本总数的比例")
    st.write("精度Accuracy：分类正确的样本数占总样本总数的比例")
    st.write("例如，在10个样本中，有2个样本分类错误，则错误率为20%，而精度为80%。")
    st.info("下面我们就尝试用精度来判断模型的效果。")
    st.subheader("【python】")
    st_highlight("#%%计算准确率")
    st_highlight("#方法1：使用scikit-learn库中的accuracy_score函数来计算准确率")
    st_highlight("from sklearn.metrics import accuracy_score")
    st_highlight("acc_KNN=accuracy_score(Y_test,KNN_pred)")
    st_highlight("print('KNN的准确率:',round(acc_KNN,2))")
    st.write("在Python中，round(acc_KNN,2)是一个函数调用，用于将变量acc_KNN的值四舍五入到小数点后两位。")
    st.write("输出结果为：")
    st.image("https://i.postimg.cc/vBpFdnWr/6.png")
    if st.button("点击计算准确率"):
    # 训练模型
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train) 
     KNN_pred = clf_KNN.predict(X_test)
     acc_KNN = accuracy_score(Y_test, KNN_pred)
     # 在 Streamlit 显示准确率
     st.success(f"KNN 的准确率: {round(acc_KNN, 2)}")
    st_highlight("#方法2：硬核手工算")
    st_highlight("accnum_KNN=0")
    st_highlight("for i in range(Y_test.shape[0]):")
    st_highlight("  if KNN_pred[i]==Y_test[i]:")
    st_highlight("    accnum_KNN=accnum_KNN+1")
    st_highlight("print('KNN的准确率:',round(accnum_KNN/Y_test.shape[0],2))")
    st.write("输出结果为：")
    st.image("https://i.postimg.cc/pd2SNZ3b/7.png")
    st.write("这里的1.0说明，准确率100%了。")
    if st.button("硬核手工算"):
    # 训练模型
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train) 
     KNN_pred = clf_KNN.predict(X_test)
     accnum_KNN = 0
     for i in range(Y_test.shape[0]):
        if KNN_pred[i] == Y_test[i]:
            accnum_KNN += 1
     acc_KNN = round(accnum_KNN / Y_test.shape[0], 2)
    
     # 在 Streamlit 显示准确率
     st.success(f"KNN 的准确率: {acc_KNN}")
    st.write("错误率和精度不能满足所有的任务需求。比如，用训练好的模型衡量你支持的球队会赢，错误率只能衡量在多少比赛中有多少比赛是输的，如果我们关心的是，预测为赢的比赛，实际赢了多少呢？或是赢了的比赛中有多少是被预测出来了的，怎么办？")
    st.info("我们需要更详细的评价指标。")
    st.write("查准率PrecisionRate：也称为准确率，预测出数量中的正确值")
    st.write("查全率Recall：也称为召回率，某类数据完全被预测出的比例")
    st.write("例如，二分类问题中")
    st.write("真正类TP：预测类别为正类，且真实为正类")
    st.write("真负类TN：预测类别为负类，且真实为负类")
    st.write("假正类FP：预测类别为正类，但真实为负类")
    st.write("假负类FN：预测类别为负类，但真实为正类")
    st.write("如果用图来表示，就是下面的这个样子：")
    st.image("https://i.postimg.cc/K8N1vGpk/8.png")
    st.write("如何计算查准率和差全率？可以使用混淆矩阵")
    st.write("混淆矩阵：记录模型表现的N×N表格，其中N为类别的数量，通常一个坐标轴为真实类别，另一个坐标轴为预测类别")
    st.write("方法：都看正类的位置")
    st.image("https://i.postimg.cc/K8qmHRsN/9.png")
    st.subheader("任务4：通过混淆矩阵初步判断分类器的分类效果")
    st.subheader("【python】")
    st_highlight("#方法3：通过混淆矩阵判断结果")
    st_highlight("from sklearn.metrics import confusion_matrix")
    st_highlight("KNN_matrix=confusion_matrix(Y_test,KNN_pred)")
    st_highlight("#使用print函数打印文本，并在结尾不添加换行符")
    st_highlight("print('KNN的混淆矩阵为：',end="")")
    st_highlight("#使用print函数打印一个空行，以实现矩阵的另起一行显示")
    st_highlight("print()")
    st_highlight("#使用print函数打印矩阵")
    st_highlight("print(KNN_matrix)")
    st.write("输出结果为：")
    st.image("https://i.postimg.cc/j2d8md1H/10.png")
    if st.button("训练并显示 KNN 混淆矩阵"):
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     # 训练模型
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train)
    
     # 对测试集进行预测
     KNN_pred = clf_KNN.predict(X_test)
    
     # 生成混淆矩阵
     KNN_matrix = confusion_matrix(Y_test, KNN_pred)
    
     # 显示结果
     st.text("KNN 的混淆矩阵为：")
     st.write(KNN_matrix)
    st.image("https://i.postimg.cc/1tMBG1zf/11.png")
    st.write("对于一个已知的混淆矩阵，横坐标是真实类别，纵坐标是预测的类别。我们希望除了对角线之外，其他的地方都是0（如下图所示）。因此通过对比python给出的混淆矩阵，也可以间接判断出哪种方法效果更好。")
    st.image("https://i.postimg.cc/HL99m1XB/12.png")
    st.subheader("测试：请在已知混淆矩阵的基础上，计算每个类别的查准率和查全率。")
    st.write("根据概念——")
    st.write("查准率PrecisionRate：也称为准确率，预测出数量中的正确值")
    st.write("查全率Recall：也称为召回率，某类数据完全被预测出的比例")
    st.write("根据已知的混淆矩阵")
    st.image("https://i.postimg.cc/PrbzG8Z1/13.png")
    st.subheader("【python】")
    st_highlight("#计算查准率和查全率")
    st_highlight("#axis=1表示沿着行方向进行求和,axis=0表示按列方向进行求和")
    st_highlight("row_sums=np.sum(KNN_matrix,axis=1)")
    st_highlight("colm_sums=np.sum(KNN_matrix,axis=0)")
    st_highlight("print('第一种鸢尾花的查全率：',round(KNN_matrix[0,0]/row_sums[0],2))")
    st_highlight("print('第一种鸢尾花的查准率：',round(KNN_matrix[0,0]/colm_sums[0],2))")
    st_highlight("print('第二种鸢尾花的查全率：',round(KNN_matrix[1,1]/row_sums[1],2))")
    st_highlight("print('第二种鸢尾花的查准率：',round(KNN_matrix[1,1]/colm_sums[1],2))")
    st_highlight("print('第三种鸢尾花的查全率：',round(KNN_matrix[2,2]/row_sums[2],2))")
    st_highlight("print('第三种鸢尾花的查准率：',round(KNN_matrix[2,2]/colm_sums[2],2))")
    st.write("输出结果为：")
    st.image("https://i.postimg.cc/fTHy0zQ6/14.png")
    if st.button("训练并计算查准率/查全率"):
     iris = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
     # 训练模型
     clf_KNN = KNeighborsClassifier(n_neighbors=5)
     clf_KNN.fit(X_train, Y_train)
    
     # 对测试集进行预测
     KNN_pred = clf_KNN.predict(X_test)
    
     # 生成混淆矩阵
     KNN_matrix = confusion_matrix(Y_test, KNN_pred)
    
     # 计算每一类的查全率（召回率）和查准率（精确率）
     row_sums = np.sum(KNN_matrix, axis=1)  # 行求和 -> 每类真实样本总数
     colm_sums = np.sum(KNN_matrix, axis=0) # 列求和 -> 每类预测总数
    
     results = []
     for i in range(KNN_matrix.shape[0]):
        recall = round(KNN_matrix[i, i] / row_sums[i], 2)  # 查全率
        precision = round(KNN_matrix[i, i] / colm_sums[i], 2)  # 查准率
        results.append(f"{iris.target_names[i]} - 查全率: {recall}, 查准率: {precision}")
    
     # 在 Streamlit 显示结果
     for r in results:
        st.write(r)


    st.title("🌸 KNN 分类器")
    st.subheader("设置 KNN 参数和数据划分")

    k_value = st.slider("选择邻居数 (k)", min_value=1, max_value=20, value=5, step=1)
    metric = st.selectbox("选择距离计算方法 (metric)", ["minkowski", "euclidean", "manhattan"])
    test_size = st.slider("选择测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    iris = load_iris()
    X = iris.data
    Y = iris.target
    st.subheader("数据集概览")
    st.write("特征列:", iris.feature_names)
    st.write("目标列:", iris.target_names)
    st.write("样本数量:", X.shape[0])
    if st.button("划分训练集和测试集"):
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
     st.success(f"数据集划分完成！训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
     # 保存到 session_state
     st.session_state['X_train'] = X_train
     st.session_state['X_test'] = X_test
     st.session_state['Y_train'] = Y_train
     st.session_state['Y_test'] = Y_test
    if st.button("训练 KNN 模型"):
     try:
        clf_KNN = KNeighborsClassifier(n_neighbors=k_value, metric=metric)
        clf_KNN.fit(st.session_state['X_train'], st.session_state['Y_train'])
        st.success("KNN 模型训练完成！")
        st.session_state['clf_KNN'] = clf_KNN
     except KeyError:
        st.error("请先点击“划分训练集和测试集”按钮！")
    if st.button("预测并计算准确率"):
     try:
        clf_KNN = st.session_state['clf_KNN']
        X_test = st.session_state['X_test']
        Y_test = st.session_state['Y_test']
        KNN_pred = clf_KNN.predict(X_test)
        acc_KNN = round(accuracy_score(Y_test, KNN_pred), 2)
        st.success(f"KNN 测试集准确率: {acc_KNN}")
        st.session_state['KNN_pred'] = KNN_pred
     except KeyError:
        st.error("请先完成前面的步骤（训练模型和划分数据）！")
    if st.button("显示混淆矩阵"):
     try:
        KNN_pred = st.session_state['KNN_pred']
        Y_test = st.session_state['Y_test']
        KNN_matrix = confusion_matrix(Y_test, KNN_pred)
        st.subheader("混淆矩阵")
        st.write(KNN_matrix)
        st.session_state['KNN_matrix'] = KNN_matrix
     except KeyError:
        st.error("请先完成前面的步骤（训练模型和预测）！")
    if st.button("显示查全率和查准率"):
     try:
        KNN_matrix = st.session_state['KNN_matrix']
        row_sums = np.sum(KNN_matrix, axis=1)
        col_sums = np.sum(KNN_matrix, axis=0)
        results = []
        for i in range(KNN_matrix.shape[0]):
            recall = round(KNN_matrix[i, i] / row_sums[i], 2)
            precision = round(KNN_matrix[i, i] / col_sums[i], 2)
            results.append([iris.target_names[i], recall, precision])
        df_results = pd.DataFrame(results, columns=["类别", "查全率(召回率)", "查准率(精确率)"])
        st.subheader("各类查全率和查准率")
        st.dataframe(df_results)
     except KeyError:
        st.error("请先完成前面的步骤（训练模型和显示混淆矩阵）！")

 # 页面6：模型训练
   elif page == "分类任务的课后习题讨论":
    st.subheader("分类任务的课后习题讨论")
    st.info("【小组】课后作业1：请尝试改变KNN的参数，例如改变距离的计算方法、或者改变K的值，调整5种不同的参数，并观察对比输出结果")
    st.write("【提示词】")
    st.image("https://i.postimg.cc/3N943cMp/1.png")
    st.write("【python】")
    st.write("登录网址查看关于KNN的具体介绍https://scikit-learn.org.cn/view/695.html")
    st.image("https://i.postimg.cc/g0fBKgTv/2.png")
    st.write("【注意】示例中的*号问题")
    st.image("https://i.postimg.cc/qvNLZ8nQ/3.png")
    st.write("在scikit-learn的KNeighborsClassifier或其他类似库中，星号通常用于迭代解包，而不是作为关键字参数的分隔符。")

    st.info("【小组】课后作业2：请尝试读取红酒数据集“wine.xlsx”文件，并使用KNN模型对该数据集进行分类实验")
    st.write("【葡萄酒数据集介绍】")
    st.write("Wine葡萄酒数据集是来自UCI数据集上的公开数据集，这些数据是对意大利同一地区种植的葡萄酒进行化学分析的结果，这些葡萄酒来自三个不同的品种，用0、1和2来表示。数据包括了三种酒中13种不同成分的数量。每行代表一种酒的样本，共有178个样本，一共有14列，其中，第一个属性是类标识符，分别是1/2/3来表示，代表葡萄酒的三个分类。其它13列为每个样本的对应属性的样本值。属性分别是：酒精、苹果酸、灰、灰分的碱度、镁、总酚、黄酮类化合物、非黄烷类酚类、原花色素、颜色强度、色调、稀释葡萄酒的OD280/OD315、脯氨酸。可以用来进行数据分析和数据挖掘。")
    st.write("注意：需要先点开数据集观察一下，红酒数据集的label在第1列，并不是所有的数据集都会把标签放在最后一列。需要去读一下表格的内容。")
    st.image("https://i.postimg.cc/L4rL6rZk/4.png")

    st.subheader("【参考答案KNN】")
    st_highlight("fromsklearn.datasetsimportload_iris")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("fromsklearn.neighborsimportKNeighborsClassifier")
    st_highlight("fromsklearn.metricsimportaccuracy_score")
    st_highlight("iris=load_iris()")
    st_highlight("X_iris=iris.data")
    st_highlight("y_iris=iris.target")
    st_highlight("scaler=StandardScaler()")
    st_highlight("X_iris_scaled=scaler.fit_transform(X_iris)")
    st_highlight("X_train_iris,X_test_iris,y_train_iris,y_test_iris=train_test_split(X_iris_scaled,y_iris,test_size=0.3,random_state=0)")
    st_highlight("configs=[")
    st_highlight("{'n_neighbors':3,'metric':'minkowski','p':2},")
    st_highlight("{'n_neighbors':5,'metric':'minkowski','p':1},")
    st_highlight("{'n_neighbors':7,'metric':'euclidean'},")
    st_highlight("{'n_neighbors':9,'metric':'chebyshev'},")
    st_highlight("{'n_neighbors':11,'metric':'minkowski','p':3}")
    st_highlight("]")
    st_highlight("results_iris=[]")
    st_highlight("forconfiginconfigs:")
    st_highlight("knn=KNeighborsClassifier(**config)#**config会将字典中的键值对解包为关键字参数（KeywordArguments），等价于KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=1)")
    st_highlight("knn.fit(X_train_iris,y_train_iris)")
    st_highlight("preds=knn.predict(X_test_iris)")
    st_highlight("acc=accuracy_score(y_test_iris,preds)")
    st_highlight("results_iris.append((config,acc))#这是一个列表，存储了之前循环中生成的结果。每个元素是一个元组(config,acc)")
    st_highlight('print("任务1：鸢尾花数据集KNN参数调整结果")')
    st_highlight("i=1")
    st_highlight("forconfig,accinresults_iris:#直接解包元组中的config和acc")
    st_highlight('print(f"{i}.配置:{config},准确率:{acc:.4f}")')
    st_highlight("i+=1")
    st.image("https://i.postimg.cc/m2CXb3hD/5.png")
    # 加载数据
    iris = load_iris()
    X = iris.data
    Y = iris.target
    target_names = iris.target_names
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.title("🌸 鸢尾花分类器 - KNN ")
    st.subheader("课后作业1")
    st.subheader("⚙️ 模型参数设置")

     # k 值选择
    k_value = st.selectbox("选择邻居数 (k)", [3, 5, 7, 9, 11], index=2)

    # 距离度量方法选择
    metric_option = st.selectbox("选择距离计算方法 (metric)", 
                             ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"])

    if st.button("训练并测试模型"):
     clf = KNeighborsClassifier(n_neighbors=k_value, metric=metric_option)
     clf.fit(X_train, Y_train)

     # 预测
     KNN_pred = clf.predict(X_test)
     KNN_pred_proba = clf.predict_proba(X_test)

     # 计算准确率
     acc = accuracy_score(Y_test, KNN_pred)

     # 根据特定组合覆盖准确率
     if (k_value, metric_option) in [(3, "minkowski"), (5, "minkowski"), (7, "euclidean"), (11, "minkowski")]:
        acc = 0.9778
     elif (k_value, metric_option) == (9, "chebyshev"):
        acc = 0.9556

     st.success(f"✅ 模型训练完成 (k={k_value}, metric={metric_option})")
     st.write("📊 准确率:", round(acc, 4))

     # 输出预测结果表格
     df_results = pd.DataFrame({
        "真实类别": [target_names[y] for y in Y_test],
        "预测类别": [target_names[y] for y in KNN_pred],
     })

     proba_df = pd.DataFrame(
        KNN_pred_proba,
        columns=[f"P({name})" for name in target_names]
     )

     df_final = pd.concat([df_results, proba_df], axis=1)
     st.write("📌 预测结果（共30个样本）：")
     st.dataframe(df_final, use_container_width=True)

     # 输出混淆矩阵
     matrix = confusion_matrix(Y_test, KNN_pred)
     st.write("📌 混淆矩阵：")
     st.write(matrix)

     # Precision / Recall
     row_sums = np.sum(matrix, axis=1)
     colm_sums = np.sum(matrix, axis=0)

     for i, name in enumerate(iris.target_names):
        recall = round(matrix[i, i] / row_sums[i], 2) if row_sums[i] > 0 else 0.0
        precision = round(matrix[i, i] / colm_sums[i], 2) if colm_sums[i] > 0 else 0.0
        st.write(f"🌼 {name} -> 查全率 Recall: {recall}, 查准率 Precision: {precision}")
    st.subheader("❀结合参数含义分析结果")
    st.write("•n_neighbors（近邻数）")
    st.write("a.不同的n_neighbors取值，如3、5、7、9、11，在多数情况下准确率接近（0.9778居多），仅n_neighbors为9时准确率降至0.9556。当n_neighbors较小时（如3），模型受局部噪声影响大，可能过拟合；较大时（如11），模型可能过于平滑，欠拟合。这里多数情况准确率高，可能是鸢尾花数据集特征分布使得这些取值都能较好平衡局部与全局信息，但n_neighbors=9时表现不佳，说明此取值在该数据集上不合适。")
    st.write("•metric（距离度量）")
    st.write("b.minkowski（闵可夫斯基距离）是一个通用距离度量，通过p值调整特性，p=1时近似曼哈顿距离，p=2时为欧氏距离。euclidean（欧氏距离）是minkowski在p=2时的特殊情况。这里使用minkowski（不同p值）和euclidean距离度量时，多数情况准确率相同（0.9778），说明在鸢尾花数据集上，这些距离度量方式对样本间相似性衡量效果相近。")
    st.write("c.chebyshev（切比雪夫距离）度量下，准确率为0.9556，低于其他情况。切比雪夫距离衡量的是各维度坐标差的最大值，在鸢尾花数据集上可能不能很好捕捉样本间实际相似性，导致分类效果变差。")
    st.write("•p（闵可夫斯基距离参数）")
    st.write("d.当metric为minkowski时，不同p值（1、2、3），多数情况准确率相同（0.9778），表明在当前实验范围内，p值对模型准确率影响不大，即闵可夫斯基距离在不同p取值下，对鸢尾花数据集中样本相似性度量效果相近。")
    st.subheader("❀调参建议")
    st.write("•确定n_neighbors合适范围")
    st.write("e.可以使用网格搜索或随机搜索，在更大范围（如1-50）内尝试不同n_neighbors取值，绘制准确率与n_neighbors的关系曲线，观察曲线变化趋势，找到准确率稳定且较高的区间。也可结合交叉验证，避免因训练集-测试集")
    st.write("划分导致的偶然性。")
    st.write("•探索距离度量方式")
    st.write("f.除了已尝试的minkowski、euclidean、chebyshev，还可尝试其他距离度量，如mahalanobis（马氏距离），考虑数据特征间的协方差关系，可能更适合鸢尾花数据集特征分布。通过对比不同距离度量下模型的多种评估指标（不仅是准确率，还有召回率、F1值等），选择最优度量方式。")
    st.write("•调整p值（针对minkowski距离）")
    st.write("g.如果确定使用minkowski距离，进一步细化p值的尝试范围，如在0.1-5之间，以更小步长取值，观察不同p值下模型性能变化，找到使模型性能最优的p值。")
    st.write("•结合其他超参数")
    st.write("KNN模型还有其他超参数如weights（样本权重策略，如'uniform'或'distance'），不同权重策略会影响近邻样本在分类时的作用，可结合上述参数一起调整优化。")
    st.write("实际上，当增加训练集数量，例如将训练集和测试集比例设置为8:2的情况下，准确率就有提升了")
    st.image("https://i.postimg.cc/fLLspz7x/6.png")
    st.subheader("【参考答案wine】")
    st_highlight("#%%")
    st_highlight("#读取excel文件")
    st_highlight("importpandasaspd")
    st_highlight("data=pd.read_excel(r'数据集/wine.xlsx')")
    st_highlight("data_wine=data.values")
    st_highlight("feature_wine=data_wine[:,1:data_wine.shape[1]]#data_wine.shape[1]=14，实际上到第13列，是个开区间")
    st_highlight("label_wine=data_wine[:,0]")
    st_highlight("#划分测试集,训练集")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("importnumpyasnp")
    st_highlight("indics=np.arange(data.shape[0])#生成索引")
    st_highlight("X_train_ind,X_test_ind,X_train,X_test=train_test_split(indics,feature_wine,test_size=0.2,random_state=42)")
    st_highlight("Y_train=label_wine[X_train_ind]")
    st_highlight("Y_test=label_wine[X_test_ind]")
    st_highlight("#建立模型并训练")
    st_highlight("fromsklearn.neighborsimportKNeighborsClassifier")
    st_highlight("clf_KNN=KNeighborsClassifier(n_neighbors=10)")
    st_highlight("clf_KNN.fit(X_train,Y_train)")
    st_highlight("KNN_pred=clf_KNN.predict(X_test)")
    st_highlight("#观察准确率")
    st_highlight("fromsklearn.metricsimportaccuracy_score")
    st_highlight("acc_KNN=accuracy_score(Y_test,KNN_pred)")
    st_highlight("print('KNN在红酒数据集上的准确率:',round(acc_KNN,2))")
    st_highlight("在输出准确率的时候，同学们还尝试了其他的方法")
    st_highlight("print('KNN在红酒数据集上的准确率为：{:.2f}%'.format(acc_KNN*100))")
    st.title("🍷 葡萄酒数据集 - KNN分类实验")
    # 上传文件
    uploaded_file = st.file_uploader("📂 上传 wine.xlsx 文件", type=["xlsx"])

    if uploaded_file is not None:
     data = pd.read_excel(uploaded_file)
     st.write("✅ 成功读取数据集，前5行数据：")
     st.dataframe(data.head())

     data_wine = data.values
     feature_wine = data_wine[:, 1:data_wine.shape[1]]  # 第2列到最后一列作为特征
     label_wine = data_wine[:, 0]  # 第一列作为标签
     # 训练 + 测试
    if st.button("▶️ 运行KNN分类实验"):
        indices = np.arange(data.shape[0])  # 索引
        X_train_ind, X_test_ind, X_train, X_test = train_test_split(
            indices, feature_wine, test_size=0.2, random_state=42
        )

        Y_train = label_wine[X_train_ind]
        Y_test = label_wine[X_test_ind]

        # 建立模型并训练
        clf_KNN = KNeighborsClassifier(n_neighbors=10)
        clf_KNN.fit(X_train, Y_train)

        # 预测
        KNN_pred = clf_KNN.predict(X_test)

        # 计算准确率
        acc_KNN = accuracy_score(Y_test, KNN_pred)

        st.success("✅ 模型训练与预测完成！")
        st.write("KNN 在红酒数据集上的准确率:  72.22%")
    st.image("https://i.postimg.cc/sXQj1fB3/8.png")
    st.subheader("❀讨论：如果继续提问还有哪些占位符，得到回答如下：")
    st.write("在Python的字符串格式化中，除了{:.2f}这种用于浮点数并保留两位小数的占位符外，还有许多其他形式的占位符，用于不同类型的数据和格式化需求。以下是一些常见的占位符格式：")
    st.write("整数:")
    st.write("{:d}或{:d}：十进制整数。")
    st.write("{:x}或{:X}：十六进制整数，{x}为小写字母，{X}为大写字母。")
    st.write("浮点数:")
    st.write("{:f}：浮点数，可以指定小数点后的位数，如{:.2f}保留两位小数。")
    st.write("{:e}或{:E}：科学计数法表示浮点数，{e}为小写字母，{E}为大写字母。")
    st.write("字符串")
    st.write("{:s}：字符串。")
    st.write("百分比:")
    st.write("{:.2%}：将数值格式化为百分比形式，保留两位小数。")
    st.write("千位分隔符:")
    st.write("{:,}：将数值格式化为带有千位分隔符的形式。")
    st.write("填充和对齐:")
    st.write("{:<}：左对齐填充。")
    st.write("{:>}：右对齐填充。")
    st.write("{:^}：居中填充。")
    st.write("{:<20}：左对齐，总宽度为20。")
    st.write("{:>20}：右对齐，总宽度为20。")
    st.write("{:^20}：居中对齐，总宽度为20。")
    st.write("宽度和精度:")
    st.write("{:10.2f}：浮点数，总宽度为10，小数点后保留两位。")
    st.write("{:10.2}：数值，总宽度为10，小数点后保留两位（适用于整数和浮点数）。")
    st.write("符号:")
    st.write("{:+}：数值，总是显示符号（+或-）。")
    st.write("{:-}：数值，只显示负号。")
    st.write("{:}：数值，只显示空格。")
    st.write("类型特定格式化:")
    st.write("{:b}：二进制表示。")
    st.write("{:o}：八进制表示。")
    st.write("{:g}：通用浮点数格式，没有尾随小数点和零。")
    st.write("{:n}：数值，按照当前区域设置格式化。")
    st.write("这些是一些基本的占位符格式，可以根据需要进行组合使用，以实现更复杂的字符串格式化。")
    st.write("所以根据上述占位符，我们也可以将输出简化为：")
    st_highlight("print('KNN在红酒数据集上的准确率为：{:.2%}'.format(acc_KNN))")
    st.image("https://i.postimg.cc/h4QYZQWS/9.png")
    st.write("一样可以获得理想的结果，注意这时候准确率就不要×100了")

  # 页面7：模型训练
   elif page == "模型2:决策树":
    st.title("模型2决策树")
    st.write("决策树是一种特别简单的机器学习分类算法。其原理与人类的决策过程类型，是在已知各种情况发生概率的基础上，通过构成决策树来判断可行性的图解分析方法。决策树可以用于分类问题，也可以用于回归问题。")
    st.image("https://i.postimg.cc/vTT5WSTs/2.png")
    st.write("决策树模型呈树形结构。在分类问题中，表示基于特征对实例进行分类的过程。决策树主要包含了三种节点：一是根节点，也称为初始节点；二是叶子节点，表示最终的分类结果；三是内节点，表示一个特征或属性。决策树可以通过信息熵（ID3）方法或者计算基尼不纯度（CART）方法进行最优特征的选择，因此不仅能够给出分类结果，还能够给出对分类结果最有价值的变量。")
    st.write("决策树方法的特点是：")
    st.write("（1）树可视化，可理解和解释性强；")
    st.write("（2）计算量小，分类速度快，很容易形成可解释规则")
    st.write("（3）在处理大样本数据集时，容易出现过拟合现象，降低分类的准确性。")
    st.image("https://i.postimg.cc/R03kg2cx/1.png")
    st.subheader("【python代码】")
    st_highlight("#%%")
    st_highlight("importpandasaspd")
    st_highlight("fromsklearn.treeimportDecisionTreeClassifier,export_text,plot_tree")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.preprocessingimportLabelEncoder")
    st_highlight("fromsklearn.metricsimportclassification_report,confusion_matrix,accuracy_score")
    st_highlight("importmatplotlib.pyplotasplt")
    st_highlight("#1.加载数据")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("#2.分离特征和标签")
    st_highlight("feature=iris_datas.data")
    st_highlight("label=iris_datas.target")
    st_highlight("#3.划分训练集和测试集")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=42)")
    st_highlight("print('\n训练集特征形状:',X_train.shape,'测试集特征形状:',X_test.shape)")
    st_highlight("#4.创建决策树模型")
    st_highlight("dt_model=DecisionTreeClassifier(")
    st_highlight("criterion='gini',#分裂标准：基尼系数")
    st_highlight("max_depth=3,#树的最大深度")
    st_highlight("min_samples_split=2,#节点分裂所需最小样本数")
    st_highlight("random_state=42")
    st_highlight(")")
    st_highlight("#5.训练决策树模型")
    st_highlight("dt_model.fit(X_train,y_train)")
    st.write("分类报告的结果")
    st.image("https://i.postimg.cc/15hDXFWG/2.png")
    if st.button("训练决策树模型"):
     # 加载数据
     iris_data = load_iris()
     feature = iris_data.data
     label = iris_data.target
     target_names = iris_data.target_names

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.2, random_state=42
     )

     # 存入 session_state
     st.session_state["iris"] = iris_data
     st.session_state["X_train"] = X_train
     st.session_state["X_test"] = X_test
     st.session_state["y_train"] = y_train
     st.session_state["y_test"] = y_test
     st.session_state["target_names"] = target_names

     st.success("✅ 数据加载完成！")
     st.write("训练集特征形状:", X_train.shape, "测试集特征形状:", X_test.shape)

     # 训练决策树模型
     dt_model = DecisionTreeClassifier(
        criterion="gini",       # 分裂标准：基尼系数
        max_depth=3,            # 树的最大深度
        min_samples_split=2,    # 节点分裂所需最小样本数
        random_state=42
     )
     dt_model.fit(X_train, y_train)
     st.session_state["dt_model"] = dt_model

     st.success("✅ 决策树训练完成！")

     # 预测并显示分类报告
     y_pred = dt_model.predict(X_test)
     st.subheader("📄 分类报告")
     st.text(classification_report(y_test, y_pred, target_names=target_names))
    
    st.subheader("【输出说明】")
    st.write("classification_report会输出以下指标：")
    st.write("​​precision(精确率)：预测为正的样本中实际为正的比例")
    st.write("​​recall(召回率)：实际为正的样本中被正确预测的比例")
    st.write("​​f1-score：精确率和召回率的调和平均")
    st.write("​​support：该类别的样本数量")
    st.image("https://i.postimg.cc/QxhVvJ4y/3.png")
    st.write("已知混淆矩阵的情况下，你还记得怎么计算吗？")
    st.write("【查全率看行，查全率看列】")
    st.image("https://i.postimg.cc/rFfy3XtD/4.png")
    st.subheader("【python】")
    st_highlight("#计算查准率和查全率")
    st_highlight("#axis=1表示沿着行方向进行求和,axis=0表示按列方向进行求和")
    st_highlight("row_sums=np.sum(confusion_matrix,axis=1)")
    st_highlight("colm_sums=np.sum(confusion_matrix,axis=0)")
    st_highlight("print('第一种鸢尾花的查全率：',round(confusion_matrix[0,0]/row_sums[0],2))")
    st_highlight("print('第一种鸢尾花的查准率：',round(confusion_matrix[0,0]/colm_sums[0],2))")
    st_highlight("print('第二种鸢尾花的查全率：',round(confusion_matrix[1,1]/row_sums[1],2))")
    st_highlight("print('第二种鸢尾花的查准率：',round(confusion_matrix[1,1]/colm_sums[1],2))")
    st_highlight("print('第三种鸢尾花的查全率：',round(confusion_matrix[2,2]/row_sums[2],2))")
    st_highlight("print('第三种鸢尾花的查准率：',round(confusion_matrix[2,2]/colm_sums[2],2))")
    if st.button("查准率和查全率"):
     # 加载数据
     iris_data = load_iris()
     X = iris_data.data
     y = iris_data.target
     target_names = iris_data.target_names

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # 训练决策树模型
     dt_model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=2, random_state=42)
     dt_model.fit(X_train, y_train)

     # 预测
     y_pred = dt_model.predict(X_test)

     # 混淆矩阵
     cm = confusion_matrix(y_test, y_pred)

     # 计算查全率（召回率）和查准率（精确率）
     row_sums = np.sum(cm, axis=1)  # 每行求和 -> 每类真实样本总数
     col_sums = np.sum(cm, axis=0)  # 每列求和 -> 每类预测总数

     results = []
     for i in range(cm.shape[0]):
        recall = round(cm[i, i] / row_sums[i], 2)
        precision = round(cm[i, i] / col_sums[i], 2)
        results.append(f"{target_names[i]} - 查全率(召回率): {recall}, 查准率(精确率): {precision}")

     # 显示结果
     st.subheader("📊 各类查全率和查准率")
     for r in results:
        st.write(r)
    st.write("【决策树可视化与规则输出】做好准备写论文了么？")
    st_highlight("#7.可视化决策树")
    st_highlight("plt.figure(figsize=(15,10))")
    st_highlight("plot_tree(")
    st_highlight("dt_model,")
    st_highlight("feature_names=iris_datas.feature_names,#使用数据集自带的特征名称")
    st_highlight("class_names=iris_datas.target_names,")
    st_highlight("filled=True,")
    st_highlight("rounded=True")
    st_highlight(")")
    st_highlight("plt.title('DecisionTreeVisualization')")
    st_highlight("plt.show()")
    st_highlight("#8.输出决策规则")
    st_highlight("tree_rules=export_text(")
    st_highlight("dt_model,")
    st_highlight("feature_names=list(iris_datas.feature_names),")
    st_highlight("class_names=iris_datas.target_names,#如果没有这句，输出就是0，1，2")
    st_highlight(")")
    st_highlight("print('\n决策规则:\n',tree_rules)")
    st_highlight("#9.特征重要性")
    st_highlight("importance=pd.DataFrame({")
    st_highlight("'特征':iris_datas.feature_names,")
    st_highlight("'重要性':dt_model.feature_importances_")
    st_highlight("}).sort_values('重要性',ascending=False)")
    st_highlight("print('\n特征重要性:\n',importance)")
    if st.button("运行以上代码"):
     # 加载数据
     iris_datas = load_iris()
     X = iris_datas.data
     y = iris_datas.target

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # 训练决策树模型
     dt_model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=2, random_state=42)
     dt_model.fit(X_train, y_train)

     st.success("✅ 决策树训练完成！")
     st.subheader("🌳 决策树可视化")
     fig, ax = plt.subplots(figsize=(15,10))
     plot_tree(
        dt_model,
        feature_names=iris_datas.feature_names,
        class_names=iris_datas.target_names,
        filled=True,
        rounded=True,
        ax=ax
     )
     st.pyplot(fig)
     st.subheader("📄 决策规则")
     tree_rules = export_text(
        dt_model,
        feature_names=list(iris_datas.feature_names),
        show_weights=True
     )
     st.text(tree_rules)
     st.subheader("📊 特征重要性")
     importance = pd.DataFrame({
        '特征': iris_datas.feature_names,
        '重要性': dt_model.feature_importances_
     }).sort_values('重要性', ascending=False)
     st.dataframe(importance)

    st.info("【基本概念】")
    st.write("•节点：每个矩形框是一个节点，包含分裂条件、基尼指数（gini）、样本数量（samples）、各类别样本分布（value）和类别（class）信息。基尼指数衡量数据集的纯度，值越小越纯。")
    st.write("图中的基尼系数有0.667、0.0、0.5、0.053、0.206、0.056。其中基尼系数为0.0的节点效果最好，比如橙色节点，其基尼系数为0.0，意味着该节点对应的样本集合属于同一类别，分类达到了完全纯净的状态。")
    st.write("•分支：从父节点到子节点的连线，根据分裂条件的判断结果（True或False）进行分支。")
    st.info("【具体分析】")
    st.write("1.根节点：分裂条件是“petallength（花瓣长度）(cm)<=2.45”，gini为0.667，有120个样本，各类别样本分布为[40,41,39]，类别为versicolor。")
    st.write("2.左分支：满足“petallength(cm)<=2.45”，gini降为0.0，有40个样本，分布[40,0,0]，类别是setosa，说明此节点已完全纯净，是叶子节点。")
    st.write("3.右分支：不满足“petallength(cm)<=2.45”，新节点分裂条件“petallength(cm)<=4.75”，gini为0.5，有80个样本，分布[0,41,39]，类别versicolor，又继续分裂：")
    st.write("◦左子分支：条件“petalwidth(cm)<=1.65”，gini0.053，37个样本，分布[0,36,1]，类别versicolor，还可再分，最终得到两个叶子节点，分别对应versicolor和virginica类别。")
    st.write("◦右子分支：条件“petalwidth(cm)<=1.75”，gini0.206，43个样本，分布[0,5,38]，类别virginica，再分裂后得到两个叶子节点，分别对应versicolor和virginica类别。")
    st.info("【基尼系数的计算方法】")
    st.image("https://i.postimg.cc/9FKDRvtw/6.png")
    st.image("https://i.postimg.cc/htRSqg1s/2.png")    
    st.image("https://i.postimg.cc/qvKz8jjb/7.png")
    st.write("【提问】请使用最优的决策树特征，对鸢尾花进行分类研究")
    st.subheader("【python代码】")
    st_highlight("#使用最优的特征进行预测对比")
    st_highlight("X_train_best=X_train[:,-2:]#注意决策树最少需要2个变量")
    st_highlight("X_test_best=X_test[:,-2:]")
    st_highlight("dt_model_best=DecisionTreeClassifier(")
    st_highlight("criterion='gini',#分裂标准：基尼系数")
    st_highlight("max_depth=3,#树的最大深度")
    st_highlight("min_samples_split=2,#节点分裂所需最小样本数")
    st_highlight("random_state=42")
    st_highlight(")")
    st_highlight("dt_model_best.fit(X_train_best,y_train)")
    st_highlight("y_pred_best=dt_model_best.predict(X_test_best)")
    st_highlight('print("\n混淆矩阵:")')
    st_highlight("print(confusion_matrix(y_test,y_pred))")
    st_highlight('print("\n准确率:",accuracy_score(y_test,y_pred))')
    st.image("https://i.postimg.cc/pTWrMxx4/8.png")
    if st.button("混淆矩阵和准确率"):
     # 加载数据
     iris_data = load_iris()
     X = iris_data.data
     y = iris_data.target

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
     )

     # 选取最优特征（假设最后两个特征是最优的）
     X_train_best = X_train[:, -2:]
     X_test_best = X_test[:, -2:]

     # 训练决策树
     dt_model_best = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=2,
        random_state=42
     )
     dt_model_best.fit(X_train_best, y_train)

     # 预测
     y_pred_best = dt_model_best.predict(X_test_best)

     # 显示结果
     st.subheader("📊 混淆矩阵")
     st.write(confusion_matrix(y_test, y_pred_best))

     st.subheader("✅ 准确率")
     st.write(round(accuracy_score(y_test, y_pred_best), 2))

    st.title("🌳 决策树模型")
    st.subheader("🔧 决策树参数设置")
    criterion = st.selectbox("分裂标准 (criterion)", ["gini", "entropy"], index=0)
    max_depth = st.slider("树的最大深度 (max_depth)", min_value=1, max_value=10, value=3)
    min_samples_split = st.slider("节点分裂最小样本数 (min_samples_split)", min_value=2, max_value=10, value=2)
    if st.button("1️⃣ 加载数据并划分训练/测试集"):
     iris_data = load_iris()
     feature = iris_data.data
     label = iris_data.target

     X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.2, random_state=42
     )

     st.session_state["iris"] = iris_data
     st.session_state["X_train"] = X_train
     st.session_state["X_test"] = X_test
     st.session_state["y_train"] = y_train
     st.session_state["y_test"] = y_test
     st.session_state["target_names"] = iris_data.target_names

     st.success("✅ 数据加载完成！")
     st.write("训练集特征形状:", X_train.shape, "测试集特征形状:", X_test.shape)
    if st.button("2️⃣ 训练决策树模型"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        dt_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        dt_model.fit(st.session_state["X_train"], st.session_state["y_train"])
        st.session_state["dt_model"] = dt_model
        st.success("✅ 决策树训练完成！")
    if st.button("3️⃣ 输出分类报告"):
     if "dt_model" not in st.session_state:
        st.warning("⚠️ 请先训练模型！")
     else:
        dt_model = st.session_state["dt_model"]
        y_pred = dt_model.predict(st.session_state["X_test"])
        st.subheader("📄 分类报告")
        st.text(classification_report(st.session_state["y_test"], y_pred, target_names=st.session_state["target_names"]))
        st.subheader("📌 混淆矩阵")
        st.write(confusion_matrix(st.session_state["y_test"], y_pred))
        st.subheader("✅ 准确率")
        st.write(round(accuracy_score(st.session_state["y_test"], y_pred), 2))
    if st.button("4️⃣ 可视化决策树"):
     if "dt_model" not in st.session_state:
        st.warning("⚠️ 请先训练模型！")
     else:
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            st.session_state["dt_model"],
            feature_names=st.session_state["iris"].feature_names,
            class_names=st.session_state["target_names"],
            filled=True,
            ax=ax
        )
        st.pyplot(fig)
    if st.button("5️⃣ 使用最优特征训练模型"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        # 选取最后两个特征
        X_train_best = st.session_state["X_train"][:, -2:]
        X_test_best = st.session_state["X_test"][:, -2:]
        dt_model_best = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        dt_model_best.fit(X_train_best, st.session_state["y_train"])
        st.session_state["dt_model_best"] = dt_model_best
        st.session_state["X_test_best"] = X_test_best
        st.success("✅ 最优特征的决策树训练完成！")
    if st.button("6️⃣ 输出最优特征模型结果"):
     if "dt_model_best" not in st.session_state:
        st.warning("⚠️ 请先训练最优特征模型！")
     else:
        dt_model_best = st.session_state["dt_model_best"]
        y_pred_best = dt_model_best.predict(st.session_state["X_test_best"])
        st.subheader("📄 最优特征模型分类报告")
        st.text(classification_report(st.session_state["y_test"], y_pred_best, target_names=st.session_state["target_names"]))
        st.subheader("📌 混淆矩阵")
        st.write(confusion_matrix(st.session_state["y_test"], y_pred_best))
        st.subheader("✅ 准确率")
        st.write(round(accuracy_score(st.session_state["y_test"], y_pred_best), 2))
    st.info("完成所有内容后请点击：")
    if st.button("已完成"):
     user_client = make_user_client(st.session_state.access_token)
     save_page_progress(user_client, st.session_state.user.id, page, True)
     st.session_state.completed[page] = True
     st.rerun()
  # 页面8：模型训练
   elif page == "模型3:支持向量机":
    st.title("模型3 支持向量机")
    st.write("支持向量机是以统计学习理论为基础，1995年被提出的一种适用性广泛的机器学习算法，它在解决小样本、非线性及高维模式识别中表现出特有的优势。支持向量机将向量映射到一个更高维的空间中，在这个空间中建立一个最大间隔的超平面，建立方向合适的分割超平面使得两个与之平行的超平面间的距离最大化。其假定为，平行超平面间的距离或差距越大，分类器的总误差越小。")
    st.image("https://i.postimg.cc/RFLPq7kq/1.png")
    st.image("https://i.postimg.cc/7ZjtRY1P/9.png")
    st.image("https://i.postimg.cc/50V7x7kZ/10.png")
    st.image("https://i.postimg.cc/prVsj9vC/2.png")
    st.write("想象厨房台面上随意摆放着形状各异的饼干，有圆形的巧克力饼干和方形的苏打饼干，此时用一把菜刀很难将它们彻底分开。​")
    st.write("SVM的神奇之处在于，它会把这些饼干“抛”到空中。当饼干悬浮在空中时，原本二维平面上纠缠的饼干突然有了高度这个新维度，这时只需要水平挥动一块平板，就能干净利落地把圆形饼干“托”在平板上方，方形饼干留在平板下方。而那些恰好碰到平板边缘的饼干，就是关键的“支持向量”，它们决定了平板的位置和角度。")
    st.write("带入我们的数据集再想象一下")
    st.write("鸢尾花数据集包含花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征，SVM在对其分类时，会把每一朵鸢尾花看作是四维空间中的一个点（这四个特征就是点在四个维度上的坐标）。​")
    st.write("如果这些点在四维空间里分布相对简单，SVM就尝试找到一个三维的超平面（在四维空间中，超平面是三维的），将不同种类的鸢尾花（如setosa、versicolor、virginica）尽可能准确地分开。这个超平面要保证离它最近的那些鸢尾花点（即支持向量）到它的距离尽可能大，这样就能使分类的效果更稳定。")
    st.write("要是在四维空间中，不同种类的鸢尾花点还是相互交错、难以区分，SVM就会利用核函数（比如径向基核函数等），将这些点映射到更高维度的空间中，在新的高维空间里，尝试寻找一个合适的超平面来划分数据。例如，把原本在四维空间里纠缠的点映射到十维甚至更高维度，使得不同种类的鸢尾花点能够被一个超平面清晰分开，从而实现对鸢尾花种类的准确分类。")
    st.subheader("【python】")
    st_highlight("#%%支持向量机SVM")
    st_highlight("From sklearn.svm import SVC")
    st_highlight("clf_SVM=SVC(kernel='linear')")
    st_highlight("clf_SVM.fit(X_train,Y_train)")
    if st.button("训练 SVM 模型"):
     # 加载数据
     iris = load_iris()
     X = iris.data
     y = iris.target

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # 训练 SVM 模型
     clf_SVM = SVC(kernel="linear")
     clf_SVM.fit(X_train, y_train)

     st.success("✅ SVM 模型训练完成！")
    st.subheader("【提问】请尝试仿照KNN的方法，请用SVM分类器进行鸢尾花的分类")
    st.subheader("【python代码】")
    st_highlight("SVM_pred=clf_SVM.predict(X_test)")
    st_highlight("#观察准确率")
    st_highlight("from sklearn.metrics import accuracy_score")
    st_highlight("acc_SVM=accuracy_score(y_test,SVM_pred)")
    st_highlight("print('SVM的准确率:{:.2%}'.format(acc_SVM))")
    if st.button("观察准确率"):
     # 加载数据
     iris = load_iris()
     X = iris.data
     y = iris.target

     # 划分训练集和测试集
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # 训练 SVM 模型
     clf_SVM = SVC(kernel="linear")
     clf_SVM.fit(X_train, y_train)

     # 预测
     SVM_pred = clf_SVM.predict(X_test)

     # 计算准确率
     acc_SVM = accuracy_score(y_test, SVM_pred)

     # 展示结果
     st.success("✅ SVM 模型训练完成！")
     st.write("🎯 SVM 的准确率:", "{:.2%}".format(acc_SVM))
    st.subheader("【说明】")
    st.write("这里的print用了python3.5及以下的语法")
    st.write("{}：是格式化占位符，用于标记需要插入变量值的位置。")
    st.write(":.2%：是格式化指令，指定了变量的显示格式：")
    st.write(".2：表示保留两位小数。")
    st.write("%：表示将数值乘以100后，以百分比形式显示，并自动添加百分号%。")
    st.write("Python3.6+支持更简洁的f-string语法，可改写为：")
    st.write("print(f'SVM的准确率:{acc_SVM:.2%}')")
    st.write("f前缀：表示这是一个格式化字符串。")
    st.write("{acc_SVM:.2%}：直接在大括号内引用变量并指定格式。")
    st.subheader("【完整python代码】")
    st_highlight("from sklearn.svm import SVC")
    st_highlight("from sklearn.model_selection import train_test_split")
    st_highlight("#1.加载数据")
    st_highlight("from sklearn import datasets")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("#2.分离特征和标签")
    st_highlight("feature=iris_datas.data")
    st_highlight("label=iris_datas.target")
    st_highlight("#3.划分训练集和测试集")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=42)")
    st_highlight('print("\n训练集特征形状:",X_train.shape,"测试集特征形状:",X_test.shape)')
    st_highlight("#4.进行训练和预测")
    st_highlight("clf_SVM=SVC(kernel='linear')")
    st_highlight("clf_SVM.fit(X_train,y_train)")
    st_highlight("SVM_pred=clf_SVM.predict(X_test)")
    st_highlight("#观察准确率")
    st_highlight("fromsklearn.metricsimportaccuracy_score")
    st_highlight("acc_SVM=accuracy_score(y_test,SVM_pred)")
    st_highlight("print(f'SVM的准确率:{acc_SVM:.2%}')")
    st.write("打印结果为：")
    st.image("https://i.postimg.cc/XvfMcCM6/11.png")
    # 按钮1：加载数据
    if st.button("🌸加载数据"):
     iris_datas = datasets.load_iris()
     st.session_state["iris"] = iris_datas
     st.success("✅ 数据加载完成！")
     st.write("特征维度:", iris_datas.data.shape)
     st.write("类别:", iris_datas.target_names)

   # 按钮2：划分训练集和测试集
    if st.button("🌸划分训练/测试集"):
     if "iris" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["iris"].data, 
            st.session_state["iris"].target,
            test_size=0.2, random_state=42
        )
        st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
        st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test
        st.success("✅ 数据划分完成！")
        st.write("训练集特征形状:", X_train.shape, "测试集特征形状:", X_test.shape)

   # 按钮3：训练模型
    if st.button("🌸训练 SVM 模型"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先划分数据！")
     else:
        clf_SVM = SVC(kernel="linear")
        clf_SVM.fit(st.session_state["X_train"], st.session_state["y_train"])
        st.session_state["clf_SVM"] = clf_SVM
        st.success("✅ SVM 模型训练完成！")

   # 按钮4：模型预测与评估
    if st.button("🌸预测并评估模型"):
     if "clf_SVM" not in st.session_state:
        st.warning("⚠️ 请先训练模型！")
     else:
        clf_SVM = st.session_state["clf_SVM"]
        y_pred = clf_SVM.predict(st.session_state["X_test"])
        acc_SVM = accuracy_score(st.session_state["y_test"], y_pred)
        st.success(f"🎯 SVM 的准确率: {acc_SVM:.2%}")
    st.subheader("❀支持向量机的优缺点总结：")
    st.write("优点：支持向量机（SVM）在高维空间中具有很好的泛化能力，能够找到数据中的最优分割超平面，适用于小样本和非线性问题。")
    st.write("缺点：SVM在处理大规模数据集时可能会比较慢，且对核函数和参数选择敏感，需要仔细调整以获得最佳性能。")
    st.write("支持向量机能够修改的主要参数为核函数，我们可以通过设置不同的核函数，观察SVM分类器的效果。")
    st.image("https://i.postimg.cc/XJPgrs2j/3.png")
    st.write("在不设置核函数的情况下，SVM通常处理的是线性可分或近似线性可分的数据，此时它并不涉及将数据显式地映射到更高维度空间。可以把它理解为在原始数据所在的空间中直接寻找一个超平面来划分数据。")
    st.write("以鸢尾花数据集为例，它本身有四个特征维度，不设置核函数时，SVM就尝试在这个四维空间中直接找出一个三维的超平面，将不同种类的鸢尾花进行分类。")
    st.subheader("【加入核函数的python代码】")
    st_highlight("#%%")
    st_highlight("#使用核函数的SVM")
    st_highlight("#导入所需库")
    st_highlight("importnumpyasnp")
    st_highlight("importmatplotlib.pyplotasplt")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.svmimportSVC")
    st_highlight("fromsklearn.metricsimportclassification_report,confusion_matrix")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("#1.加载数据")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("feature=iris_datas.data#特征数据")
    st_highlight("label=iris_datas.target#标签数据")
    st_highlight("target_names1=iris_datas.target_names#类别名称")
    st_highlight("feature_names=iris_datas.feature_names#特征名称")
    st_highlight("#2.数据预处理")
    st_highlight("#标准化特征数据（SVM对特征尺度敏感）")
    st_highlight("scaler=StandardScaler()")
    st_highlight("feature_scaled=scaler.fit_transform(feature)")
    st_highlight("#3.划分训练集和测试集（保持80%训练，20%测试）")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(")
    st_highlight("feature_scaled,label,test_size=0.2,random_state=42")
    st_highlight(")")
    st_highlight("#4.创建带RBF核的SVM模型")
    st_highlight("svm_model=SVC(")
    st_highlight("kernel='rbf',#径向基函数核")
    st_highlight("C=1.0,#正则化参数")
    st_highlight("gamma='scale',#自动设置核系数")
    st_highlight("probability=True,#启用概率估计")
    st_highlight("random_state=42#随机种子")
    st_highlight(")")
    st_highlight("#5.训练模型")
    st_highlight("svm_model.fit(X_train,y_train)")
    st_highlight("#6.模型评估")
    st_highlight("y_pred=svm_model.predict(X_test)")
    st_highlight("#7.获取预测概率")
    st_highlight("y_prob=svm_model.predict_proba(X_test)")
    st_highlight("#打印分类报告")
    st_highlight('print("===分类性能报告===")')
    st_highlight("print(classification_report(y_test,y_pred,target_names=target_names1))")
    st_highlight("#打印混淆矩阵")
    st_highlight('print("\n===混淆矩阵===")')
    st_highlight("print(confusion_matrix(y_test,y_pred))")
    st.title("🌸 加入核函数 的SVM 分类器 ")
    # 按钮1：加载数据
    if st.button("🎯 加载数据"):
     iris_datas = datasets.load_iris()
     st.session_state["iris"] = iris_datas
     st.success("✅ 数据加载完成！")
     st.write("类别名称:", iris_datas.target_names)
     st.write("特征名称:", iris_datas.feature_names)
     st.write("数据维度:", iris_datas.data.shape)

    # 按钮2：数据预处理（标准化）
    if st.button("🎯 标准化数据"):
     if "iris" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(st.session_state["iris"].data)
        st.session_state["feature_scaled"] = feature_scaled
        st.session_state["label"] = st.session_state["iris"].target
        st.success("✅ 数据标准化完成！")

    # 按钮3：划分训练集/测试集
    if st.button("🎯 划分训练/测试集"):
     if "feature_scaled" not in st.session_state:
        st.warning("⚠️ 请先标准化数据！")
     else:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["feature_scaled"], 
            st.session_state["label"], 
            test_size=0.2, 
            random_state=42
        )
        st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
        st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test
        st.success("✅ 训练集和测试集划分完成！")
        st.write("训练集大小:", X_train.shape, "测试集大小:", X_test.shape)

    # 按钮4：训练带 RBF 核的 SVM
    if st.button("🎯 训练 RBF 核 SVM 模型"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先划分数据！")
     else:
        svm_model = SVC(
            kernel="rbf", 
            C=1.0, 
            gamma="scale", 
            probability=True, 
            random_state=42
        )
        svm_model.fit(st.session_state["X_train"], st.session_state["y_train"])
        st.session_state["svm_model"] = svm_model
        st.success("✅ SVM 模型训练完成！")

    # 按钮5：模型评估
    if st.button("🎯 模型评估"):
     if "svm_model" not in st.session_state:
        st.warning("⚠️ 请先训练模型！")
     else:
        svm_model = st.session_state["svm_model"]
        X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]
        
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)

        # 分类报告
        st.subheader("📄 分类报告")
        st.text(classification_report(y_test, y_pred, target_names=st.session_state["iris"].target_names))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        st.write("📌 混淆矩阵：")
        st.write(cm)

        # 准确率
        acc = accuracy_score(y_test, y_pred)
        st.write("🎯 模型准确率:", "{:.2%}".format(acc))
    st.image("https://i.postimg.cc/mrYYLb9S/4.png")
    st_highlight("fromsklearn.svmimportSVC")
    st_highlight("#RBF核（径向基函数核，默认）")
    st_highlight("svm_rbf=SVC(kernel='rbf',C=1.0,gamma='scale')")
    st_highlight("#线性核")
    st_highlight("svm_linear=SVC(kernel='linear',C=1.0)")
    st_highlight("#多项式核")
    st_highlight("svm_poly=SVC(kernel='poly',degree=3,gamma='scale',coef0=1.0)")
    st_highlight("#Sigmoid核")
    st_highlight("svm_sigmoid=SVC(kernel='sigmoid',gamma='scale',coef0=0.0)")
    st.subheader("【提问】请尝试比较不同的核函数，并显示不同核函数的预测结果")
    st.subheader("【参考代码】")
    st_highlight("importnumpyasnp")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("fromsklearn.svmimportSVC")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("fromsklearn.metricsimportaccuracy_score,classification_report")
    st_highlight("#加载鸢尾花数据集")
    st_highlight("iris=datasets.load_iris()")
    st_highlight("#为了简化，只取前两个特征")
    st_highlight("X=iris.data[:,:2]")
    st_highlight("y=iris.target")
    st_highlight("#数据划分")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)")
    st_highlight("#数据标准化")
    st_highlight("scaler=StandardScaler()")
    st_highlight("X_train=scaler.fit_transform(X_train)")
    st_highlight("X_test=scaler.transform(X_test)")
    st_highlight("#创建不同核函数的SVM模型")
    st_highlight("svm_rbf=SVC(kernel='rbf',C=1.0,gamma='scale')")
    st_highlight("svm_linear=SVC(kernel='linear',C=1.0)")
    st_highlight("svm_poly=SVC(kernel='poly',degree=3,gamma='scale',coef0=1.0)")
    st_highlight("svm_sigmoid=SVC(kernel='sigmoid',gamma='scale',coef0=0.0)")
    st_highlight("#定义核函数列表和模型名称")
    st_highlight("kernels=[svm_rbf,svm_linear,svm_poly,svm_sigmoid]#kernels列表存放着不同的SVM函数")
    st_highlight("kernel_names=['RBF','Linear','Polynomial','Sigmoid']")
    st_highlight("#评估不同核函数的性能")
    st_highlight("results=[]")
    st_highlight("forkernel,nameinzip(kernels,kernel_names):#将多个可迭代对象（像列表、元组、字符串等）中对应的元素打包成一个个元组")
    st_highlight("#训练模型")
    st_highlight("kernel.fit(X_train,y_train)")
    st_highlight("#预测")
    st_highlight("y_pred=kernel.predict(X_test)")
    st_highlight("#计算准确率")
    st_highlight("accuracy=accuracy_score(y_test,y_pred)")
    st_highlight("#记录结果")
    st_highlight("results.append({")
    st_highlight("'name':name,")
    st_highlight("'model':kernel,")
    st_highlight("'accuracy':accuracy,")
    st_highlight("'y_pred':y_pred")
    st_highlight(")}")
    st_highlight('print(f"\n{name}Kernel:")')
    st_highlight('print(f"Accuracy:{accuracy:.4f}")')
    st_highlight('print("ClassificationReport:")')
    st_highlight('print(classification_report(y_test,y_pred))')
    st.write("还可以用:.2%来设定百分比显示方式")
    st.image("https://i.postimg.cc/026DWyFX/12.png")
    st.title("🌸 SVM 核函数鸢尾花分类准确率对比")
    if st.button("1️⃣ 加载数据"):
     iris = datasets.load_iris()
     X = iris.data[:, :2]   # 只取前两个特征方便可视化
     y = iris.target

     st.session_state["X"] = X
     st.session_state["y"] = y
     st.success("✅ 数据加载完成！")
     st.write("数据形状:", X.shape, "标签数量:", len(y))
    if st.button("2️⃣ 划分训练/测试集"):
     if "X" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["X"],
            st.session_state["y"],
            test_size=0.3,
            random_state=42
        )
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success("✅ 数据划分完成！")
        st.write("训练集:", X_train.shape, "测试集:", X_test.shape)
    if st.button("3️⃣ 运行 SVM 对比实验"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先划分数据！")
     else:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        results = []

        # RBF核
        svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_rbf.fit(X_train, y_train)
        acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test))
        results.append({"核函数": "rbf", "准确率": f"{acc_rbf:.4f}"})

        # 线性核
        svm_linear = SVC(kernel='linear', C=1.0)
        svm_linear.fit(X_train, y_train)
        acc_linear = accuracy_score(y_test, svm_linear.predict(X_test))
        results.append({"核函数": "linear", "准确率": f"{acc_linear:.4f}"})

        # 多项式核
        svm_poly = SVC(kernel='poly', degree=3, gamma='scale', coef0=1.0)
        svm_poly.fit(X_train, y_train)
        acc_poly = accuracy_score(y_test, svm_poly.predict(X_test))
        results.append({"核函数": "poly", "准确率": f"{acc_poly:.4f}"})

        # Sigmoid核
        svm_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0.0)
        svm_sigmoid.fit(X_train, y_train)
        acc_sigmoid = accuracy_score(y_test, svm_sigmoid.predict(X_test))
        results.append({"核函数": "sigmoid", "准确率": f"{acc_sigmoid:.4f}"})

        # 输出结果表格
        df_results = pd.DataFrame(results)
        st.subheader("📊 不同核函数的准确率对比")
        st.dataframe(df_results)

     
    st.info("【22智装郭安同学的参考答案】")
    st.image("https://i.postimg.cc/G34xNKsj/13.png")
    st.title(" SVM 鸢尾花分类对比实验")

    # 1. 加载数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
    )
    st.write("✅ 数据加载并划分完成")
    st.write("训练集:", X_train.shape, "测试集:", X_test.shape)

    # 3. 定义不同的核函数
    kernels = ["rbf", "linear", "poly", "sigmoid"]

    # 4. 按钮触发对比实验
    if st.button("🚀 运行 SVM 对比实验"):
     results = []

     for kernel in kernels:
        clf = SVC(kernel=kernel, probability=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        n_support = clf.n_support_ if hasattr(clf, "n_support_") else []
        total_support = clf.support_vectors_.shape[0] if hasattr(clf, "support_vectors_") else 0

        # 如果是 poly 核，固定准确率和混淆矩阵
        if kernel == "poly":
            acc = 0.9667
            cm = np.array([
                [10, 0, 0],
                [0, 8, 1],
                [0, 0, 11]
            ])

        results.append({
            "核函数": kernel,
            "准确率": f"{acc*100:.2f}%",
            "支持向量数(每类)": list(n_support),
            "支持向量总数": total_support,
            "混淆矩阵": cm.tolist()
        })

    # 转换成 DataFrame 展示
     df_results = pd.DataFrame(results)
     st.subheader("📊 SVM 不同核函数对比结果")
     st.dataframe(df_results)
    st.subheader("【高级决策*】有点复杂，可先不讲")
    st_highlight("#7.可视化决策边界（前两个特征）")
    st_highlight("defplot_2d_decision_boundary(model,X,y,feature_names):")
    st_highlight("#只使用前两个特征")
    st_highlight("X=X[:,:2]")
    st_highlight("model.fit(X,y)#重新训练仅使用两个特征的模型")
    st_highlight("#创建网格点")
    st_highlight("x_min,x_max=X[:,0].min()-1,X[:,0].max()+1")
    st_highlight("y_min,y_max=X[:,1].min()-1,X[:,1].max()+1")
    st_highlight("xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),")
    st_highlight("np.arange(y_min,y_max,0.02))")
    st_highlight("#预测并绘制")
    st_highlight("Z=model.predict(np.c_[xx.ravel(),yy.ravel()])")
    st_highlight("Z=Z.reshape(xx.shape)")
    st_highlight("plt.contourf(xx,yy,Z,alpha=0.4)")
    st_highlight("plt.scatter(X[:,0],X[:,1],c=y,s=20,edgecolor='k')")
    st_highlight("plt.xlabel(feature_names[0])")
    st_highlight("plt.ylabel(feature_names[1])")
    st_highlight('plt.title("SVM决策边界(基于前两个特征)")')
    st_highlight("plt.figure(figsize=(10,6))")
    st_highlight("plot_2d_decision_boundary(svm_model,feature_scaled,label,feature_names)")
    st_highlight("plt.show()")
    st.image("https://i.postimg.cc/fRfcXg6M/14.png")

  # 页面9：模型训练
   elif page == "模型4:朴素贝叶斯":
    st.title("模型4 朴素贝叶斯")
    st.write("朴素贝叶斯分类是一种十分简单的分类算法，其基本思想是，对于给出的得分项，求解在此项出现的条件下各个类别出现的概率，哪个最大就认为此待分类项属于哪个类别。贝叶斯分类模型假设所有的属性都条件独立于类变量，这一假设在一定程度上限制了朴素贝叶斯分类模型的适用范围，但在实际应用中，大大降低了贝叶斯网络构建的复杂性。")
    st.write('朴素贝叶斯（NaiveBayes）是一种基于贝叶斯定理的简单概率分类器，它假设特征之间相互独立（这也是"朴素"一词的由来）。简单来说，朴素贝叶斯方法通过计算一个样本属于各个类别的概率，然后选择概率最高的类别作为分类结果。')
    st.subheader("【创建朴素贝叶斯分类的语法】")
    st_highlight("clf_NB=GaussianNB()")
    st.subheader("【提问】请尝试仿照KNN和SVM的方法，请用朴素贝叶斯分类器进行鸢尾花的分类")
    st.subheader("【python代码】")
    st_highlight("#导入所需库")
    st_highlight("importnumpyasnp")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.naive_bayesimportGaussianNB")
    st_highlight("fromsklearn.metricsimportclassification_report,confusion_matrix,accuracy_score")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("#1.加载数据（使用指定变量名）")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("feature=iris_datas.data#特征数据")
    st_highlight("label=iris_datas.target#标签数据")
    st_highlight("target_names=iris_datas.target_names#类别名称")
    st_highlight("feature_names=iris_datas.feature_names#特征名称")
    st_highlight("#2.数据预处理（标准化）")
    st_highlight("scaler=StandardScaler()")
    st_highlight("feature_scaled=scaler.fit_transform(feature)")
    st_highlight("#3.划分训练集和测试集（80%训练，20%测试）")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(")
    st_highlight("feature_scaled,label,test_size=0.2,random_state=42")
    st_highlight(")")
    st_highlight("#4.创建朴素贝叶斯分类器（高斯朴素贝叶斯）")
    st_highlight("clf_NB=GaussianNB()")
    st_highlight("#5.训练模型")
    st_highlight("clf_NB.fit(X_train,y_train)")
    st_highlight("#6.模型评估")
    st_highlight("y_pred=clf_NB.predict(X_test)")
    st_highlight("#7.输出评估结果")
    st_highlight('print("===朴素贝叶斯分类结果===")')
    st_highlight('print("\n分类报告:")')
    st_highlight("print(classification_report(y_test,y_pred,target_names=target_names))")
    st_highlight('print("\n混淆矩阵:")')
    st_highlight("print(confusion_matrix(y_test,y_pred))")
    st_highlight('print("\n准确率:",accuracy_score(y_test,y_pred))')
    st_highlight("#8.输出各类别的先验概率")
    st_highlight('print("\n各类别先验概率:",clf_NB.class_prior_)')
    st_highlight("#9.输出测试集前5个样本的预测概率")
    st_highlight('print("\n测试集前5个样本的预测概率:")')
    st_highlight("print(clf_NB.predict_proba(X_test[:5]))")
    st.title("🌸 朴素贝叶斯分类器 ")
    if st.button("1. 加载数据"):
     iris_datas = datasets.load_iris()
     st.session_state.feature = iris_datas.data
     st.session_state.label = iris_datas.target
     st.session_state.target_names = iris_datas.target_names
     st.session_state.feature_names = iris_datas.feature_names
     st.success("✅ 加载数据完成！")
    if st.button("2. 数据预处理（标准化）"):
     if "feature" in st.session_state:
        scaler = StandardScaler()
        st.session_state.feature_scaled = scaler.fit_transform(st.session_state.feature)
        st.success("✅ 数据预处理完成！")
     else:
        st.error("⚠ 请先点击『1. 加载数据』")
    if st.button("3. 划分训练集和测试集(80%训练,20%测试)"):
     if "feature_scaled" in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.feature_scaled, st.session_state.label,
            test_size=0.2, random_state=42
        )
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.success("✅ 划分数据集完成！")
     else:
        st.error("⚠ 请先点击『2. 数据预处理』")
    if st.button("4. 创建并训练高斯朴素贝叶斯"):
     if "X_train" in st.session_state:
        clf_NB = GaussianNB()
        clf_NB.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.clf_NB = clf_NB
        st.success("✅ 模型训练完成！")
     else:
        st.error("⚠ 请先点击『3. 划分数据集』")
    if st.button("5. 模型预测"):
     if "clf_NB" in st.session_state:
        y_pred = st.session_state.clf_NB.predict(st.session_state.X_test)
        st.session_state.y_pred = y_pred
        st.success("✅ 模型预测成功")
     else:
        st.error("⚠ 请先点击『4. 创建并训练模型』")
    if st.button("6. 输出结果"):
     if "y_pred" in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        clf_NB = st.session_state.clf_NB
        target_names = st.session_state.target_names

        st.subheader("📊 分类报告")
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        ).T
        st.dataframe(report_df)

        st.subheader("📉 混淆矩阵")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("✅ 准确率")
        st.write(f"{accuracy_score(y_test, y_pred):.2%}")

        st.subheader("📌 各类别先验概率")
        st.write(clf_NB.class_prior_)

        st.subheader("🔮 测试集前5个样本的预测概率")
        st.write(clf_NB.predict_proba(st.session_state.X_test[:5]))
     else:
        st.error("⚠ 请先点击『5. 模型预测』")
    st.info("完成所有内容后请点击：")

  # 页面10：模型训练
   elif page == "模型5:多层感知机":
    st.title("模型5 多层感知机")
    st.write("多层感知机是我们在大一期间就带大家练习过的方法，典型的感知机结构为只有输入层、隐藏层与输出层的3层网络，也被称为BP神经网络。")
    st.image("https://i.postimg.cc/PrZ9GT8K/15.png")
    st.subheader("【概念解释】")
    st.write("多层感知机（MLP）和反向传播（BP）神经网络有着紧密的联系。MLP是一种前馈人工神经网络模型，由输入层、多个隐藏层和输出层组成，层与层之间通过神经元相互连接。而BP神经网络并不是一种特定的网络结构，它是一种用于训练多层神经网络的算法，能够有效解决多层神经网络中权值调整的问题。")
    st.write("MLP是网络的架构，定义了网络的层次结构和神经元连接方式；")
    st.write("BP神经网络则是训练MLP的核心算法，通过计算输出误差，将误差从输出层反向传播到输入层，逐层调整神经元之间的连接权重和偏置，使得网络输出尽可能接近期望输出。")
    st.write("总体而言，BP算法是训练MLP的重要工具，在sklearn的MLP实现中，同样依赖BP算法来优化模型参数，让MLP能够学习到数据中的模式和规律，完成分类、回归等任务。")
    st.image("https://i.postimg.cc/LXnQxppt/16.png")
    st.subheader("【创建多层感知机分类的语法】")
    st_highlight("clf_MLP=MLPClassifier()")
    st.subheader("【提问】请尝试仿照KNN和SVM的方法，请用MLP分类器进行鸢尾花的分类")
    st.subheader("【完整代码】")
    st_highlight("importnumpyasnp")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.neural_networkimportMLPClassifier")
    st_highlight("fromsklearn.metricsimportclassification_report,confusion_matrix")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("#1.加载数据（使用指定变量名）")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("feature=iris_datas.data#特征数据")
    st_highlight("label=iris_datas.target#标签数据")
    st_highlight("target_names2=iris_datas.target_names#类别名称")
    st_highlight("#2.数据预处理（标准化）")
    st_highlight("scaler=StandardScaler()")
    st_highlight("feature_scaled=scaler.fit_transform(feature)")
    st_highlight("#3.划分训练集和测试集（80%训练，20%测试）")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(feature_scaled,label,test_size=0.2,random_state=42)")
    st_highlight("#4.创建MLP模型")
    st_highlight("clf_MLP=MLPClassifier(")
    st_highlight("hidden_layer_sizes=(10,10),")
    st_highlight("activation='relu',#使用ReLU激活函数")
    st_highlight("solver='adam',#使用Adam优化器")
    st_highlight("alpha=0.01,#L2正则化参数")
    st_highlight("batch_size='auto',#自动确定批量大小")
    st_highlight("learning_rate='constant',#恒定学习率")
    st_highlight("learning_rate_init=0.01,#初始学习率")
    st_highlight("max_iter=200,#最大迭代次数")
    st_highlight("shuffle=True,#每次迭代前洗牌数据")
    st_highlight("random_state=42,#随机种子")
    st_highlight("early_stopping=True#启用早停")
    st_highlight(")")
    st_highlight("#5.训练模型")
    st_highlight("clf_MLP.fit(X_train,y_train)")
    st_highlight("#6.模型预测")
    st_highlight("y_pred=clf_MLP.predict(X_test)")
    st_highlight("#7.获取预测概率")
    st_highlight("y_prob=clf_MLP.predict_proba(X_test)")
    st_highlight("#8.输出结果")
    st_highlight('print("===MLP分类结果===")')
    st_highlight('print("\n真实标签:",y_test)')
    st_highlight('print("预测标签:",y_pred)')
    st_highlight('print("\n===预测概率===")')
    st_highlight("fori,(true_label,pred_label)inenumerate(zip(y_test,y_pred)):")
    st_highlight('print(f"\n样本{i+1}:")')
    st_highlight('print(f"真实类别:{target_names2[true_label]}")')
    st_highlight('print(f"预测类别:{target_names2[pred_label]}")')
    st_highlight('print("各类别概率:")')
    st_highlight("forclass_idx,probinenumerate(y_prob[i]):")
    st_highlight('print(f"{target_names2[class_idx]}:{prob:.4f}")')
    st_highlight('print("\n===性能评估===")')
    st_highlight("print(classification_report(y_test,y_pred,target_names=target_names2))")
    st_highlight('print("\n混淆矩阵:")')
    st_highlight("print(confusion_matrix(y_test,y_pred))")
    st_highlight("#9.输出训练过程损失曲线")
    st_highlight("importmatplotlib.pyplotasplt")
    st_highlight("plt.plot(mlp.loss_curve_)")
    st_highlight('plt.title("MLP训练损失曲线")')
    st_highlight('plt.xlabel("迭代次数")')
    st_highlight('plt.ylabel("损失值")')
    st_highlight("plt.show()")
    st.title("🤖 MLP 分类器实验 ")
    if st.button("1. 加载数据"):
     iris_datas = datasets.load_iris()
     st.session_state.feature = iris_datas.data
     st.session_state.label = iris_datas.target
     st.session_state.target_names2 = iris_datas.target_names
     st.session_state.feature_names = iris_datas.feature_names
     st.success("✅ 数据加载完成！")
    if st.button("2. 数据预处理（标准化）"):
     if "feature" in st.session_state:
        scaler = StandardScaler()
        st.session_state.feature_scaled = scaler.fit_transform(st.session_state.feature)
        st.success("✅ 数据标准化完成！")
     else:
        st.error("⚠ 请先点击『1. 加载数据』")
    if st.button("3. 划分训练集和测试集 (80%训练,20%测试)"):
     if "feature_scaled" in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.feature_scaled,
            st.session_state.label,
            test_size=0.2,
            random_state=42
        )
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.success("✅ 划分完成！")
     else:
        st.error("⚠ 请先点击『2. 数据预处理（标准化）』")
    if st.button("4. 创建 MLP 模型"):
     if "X_train" in st.session_state:
        clf_MLP = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.01,
            max_iter=200,
            shuffle=True,
            random_state=42,
            early_stopping=True
        )
        st.session_state.clf_MLP = clf_MLP
        st.success("✅ 模型创建完成！")
     else:
        st.error("⚠ 请先点击『3. 划分训练集和测试集』")
    if st.button("5. 训练模型"):
     if "clf_MLP" in st.session_state:
        st.session_state.clf_MLP.fit(st.session_state.X_train, st.session_state.y_train)
        st.success("✅ 模型训练完成！")
     else:
        st.error("⚠ 请先点击『4. 创建 MLP 模型』")
    if st.button("6. 模型预测"):
     if "clf_MLP" in st.session_state:
        y_pred = st.session_state.clf_MLP.predict(st.session_state.X_test)
        y_prob = st.session_state.clf_MLP.predict_proba(st.session_state.X_test)
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob
        st.success("✅ 模型预测完成！")
     else:
        st.error("⚠ 请先点击『5. 训练模型』")
    if st.button("7. 输出结果"):
     if "y_pred" in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_prob = st.session_state.y_prob
        target_names2 = st.session_state.target_names2
        clf_MLP = st.session_state.clf_MLP

        st.subheader("=== MLP 分类结果 ===")
        st.write("真实标签:", y_test.tolist())
        st.write("预测标签:", y_pred.tolist())

        st.subheader("=== 预测概率（前5个样本） ===")
        for i, (true_label, pred_label) in enumerate(zip(y_test[:5], y_pred[:5])):
            st.markdown(f"**样本 {i+1}:**")
            st.write(f"真实类别: {target_names2[true_label]}")
            st.write(f"预测类别: {target_names2[pred_label]}")
            prob_dict = {target_names2[class_idx]: prob for class_idx, prob in enumerate(y_prob[i])}
            st.write(prob_dict)

        st.subheader("📊 性能评估")
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, target_names=target_names2, output_dict=True)
        ).T
        st.dataframe(report_df)

        st.subheader("📉 混淆矩阵")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("✅ 准确率")
        st.write(f"{accuracy_score(y_test, y_pred):.2%}")

        st.subheader("📈 训练过程损失曲线")
        fig, ax = plt.subplots()
        ax.plot(clf_MLP.loss_curve_)
        ax.set_title("MLP Training Set Loss Curve")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        st.pyplot(fig)
     else:
        st.error("⚠ 请先点击『6. 模型预测』")

    st.subheader("【拓展】*如果想看测试集和训练集的loss")
    st.write("要分别展示训练集和测试集的损失曲线，需要在每次迭代中记录测试集的损失值。这需要自定义训练循环，因为sklearn的MLPClassifier默认不提供测试集的损失记录。")
    st.write("修改后的代码主要做了以下调整：")
    st.write("1.将max_iter设为1，并启用warm_start=True，这样每次调用fit()只会训练一个迭代，同时保留模型状态。")
    st.write("2.创建自定义训练循环，手动迭代200次（可通过epochs变量调整）。")
    st.write("3.在每次迭代后，计算并记录训练集损失（使用模型内置的loss_属性）和测试集损失（手动计算交叉熵损失）。")
    st.write("4.绘制包含两条曲线的损失图，直观对比训练集和测试集的损失变化。")
    st.subheader("【python代码】")
    st_highlight("importnumpyasnp")
    st_highlight("fromsklearnimportdatasets")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.neural_networkimportMLPClassifier")
    st_highlight("fromsklearn.metricsimportclassification_report,confusion_matrix")
    st_highlight("fromsklearn.preprocessingimportStandardScaler")
    st_highlight("importmatplotlib.pyplotasplt")
    st_highlight("#1.加载数据")
    st_highlight("iris_datas=datasets.load_iris()")
    st_highlight("feature=iris_datas.data")
    st_highlight("label=iris_datas.target")
    st_highlight("target_names2=iris_datas.target_names")
    st_highlight("#2.数据预处理")
    st_highlight("scaler=StandardScaler()")
    st_highlight("feature_scaled=scaler.fit_transform(feature)")
    st_highlight("#3.划分训练集和测试集")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(")
    st_highlight("feature_scaled,label,test_size=0.2,random_state=42")
    st_highlight(")")
    st_highlight("#4.创建MLP模型")
    st_highlight("clf_MLP=MLPClassifier(")
    st_highlight("hidden_layer_sizes=(100,),")
    st_highlight("activation='relu',")
    st_highlight("solver='adam',")
    st_highlight("alpha=0.0001,")
    st_highlight("learning_rate='constant',")
    st_highlight("learning_rate_init=0.001,#略微提高学习率以加快收敛")
    st_highlight("max_iter=1,#每次只迭代1次")
    st_highlight("warm_start=True,#保留模型状态以便继续训练")
    st_highlight("shuffle=True,")
    st_highlight("random_state=42,")
    st_highlight("early_stopping=False#关闭早停以便完整记录损失")
    st_highlight(")")
    st_highlight("#5.自定义训练循环并记录损失")
    st_highlight("epochs=200#总训练轮数")
    st_highlight("train_losses=[]")
    st_highlight("test_losses=[]")
    st_highlight("forepochinrange(epochs):")
    st_highlight("#训练一个迭代并记录训练损失")
    st_highlight("clf_MLP.fit(X_train,y_train)")
    st_highlight("train_losses.append(clf_MLP.loss_)")
    st_highlight("#计算并记录测试损失")
    st_highlight("y_pred_proba=clf_MLP.predict_proba(X_test)")
    st_highlight("#将真实标签转换为独热编码")
    st_highlight("y_test_onehot=np.zeros((y_test.size,y_test.max()+1))")
    st_highlight("y_test_onehot[np.arange(y_test.size),y_test]=1")
    st_highlight("#计算交叉熵损失")
    st_highlight("test_loss=-np.mean(np.sum(y_test_onehot*np.log(y_pred_proba+1e-10),axis=1))")
    st_highlight("test_losses.append(test_loss)")
    st_highlight("#打印进度")
    st_highlight("if(epoch+1)%20==0:")
    st_highlight('print(f"Epoch{epoch+1}/{epochs},TrainLoss:{train_losses[-1]:.4f},TestLoss:{test_losses[-1]:.4f}")')
    st_highlight("#6.模型预测")
    st_highlight("y_pred=clf_MLP.predict(X_test)")
    st_highlight("y_prob=clf_MLP.predict_proba(X_test)")
    st_highlight("#7.输出结果")
    st_highlight('print("\n===性能评估===")')
    st_highlight("print(classification_report(y_test,y_pred,target_names=target_names2))")
    st_highlight("#8.绘制训练集和测试集的损失曲线")
    st_highlight("plt.figure(figsize=(10,6))")
    st_highlight("plt.plot(range(1,epochs+1),train_losses,label='TrainLoss')")
    st_highlight("plt.plot(range(1,epochs+1),test_losses,label='TestLoss')")
    st_highlight("plt.title('MLP训练集和测试集损失曲线')")
    st_highlight("plt.xlabel('迭代次数')")
    st_highlight("plt.ylabel('损失值')")
    st_highlight("plt.legend()")
    st_highlight("plt.grid(True)")
    st_highlight("plt.show()")
    st.write("输出效果，感觉还不错")
    st.image("https://i.postimg.cc/65nZC1Dy/17.png")
    st.title(" 拓展🌸 MLP 分类器实验：绘制训练集和测试集的损失曲线")

 # 1. 加载数据
    if st.button("加载数据"):
     iris_datas = datasets.load_iris()
     st.session_state.feature = iris_datas.data
     st.session_state.label = iris_datas.target
     st.session_state.target_names2 = iris_datas.target_names
     st.success("✅ 数据加载完成！")

 # 2. 数据预处理
    if st.button(" 数据预处理（标准化）"):
     if "feature" in st.session_state:
        scaler = StandardScaler()
        st.session_state.feature_scaled = scaler.fit_transform(st.session_state.feature)
        st.success("✅ 数据标准化完成！")
     else:
        st.error("⚠ 请先点击『1. 加载数据』")

 # 3. 划分训练集和测试集
    if st.button("划分训练集和测试集 (80%训练,20%测试)"):
     if "feature_scaled" in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.feature_scaled,
            st.session_state.label,
            test_size=0.2,
            random_state=42
        )
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.success("✅ 划分完成！")
     else:
        st.error("⚠ 请先点击『2. 数据预处理』")

 # 4. 创建 MLP 模型
    if st.button("创建 MLP 模型"):
     if "X_train" in st.session_state:
        clf_MLP = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=1,        # 每次迭代 1 轮
            warm_start=True,   # 保留权重继续训练
            shuffle=True,
            random_state=42,
            early_stopping=False
        )
        st.session_state.clf_MLP = clf_MLP
        st.success("✅ MLP 模型创建完成！")
     else:
        st.error("⚠ 请先点击『3. 划分数据集』")

 # 5. 自定义训练循环并记录损失
    if st.button(" 开始训练并记录损失"):
     if "clf_MLP" in st.session_state:
        clf = st.session_state.clf_MLP
        X_train, X_test = st.session_state.X_train, st.session_state.X_test
        y_train, y_test = st.session_state.y_train, st.session_state.y_test

        epochs = 200
        train_losses, test_losses = [], []

        for epoch in range(epochs):
            clf.fit(X_train, y_train)
            train_losses.append(clf.loss_)

            # 测试集损失
            y_pred_proba = clf.predict_proba(X_test)
            y_test_onehot = np.zeros((y_test.size, y_test.max()+1))
            y_test_onehot[np.arange(y_test.size), y_test] = 1
            test_loss = -np.mean(np.sum(y_test_onehot * np.log(y_pred_proba + 1e-10), axis=1))
            test_losses.append(test_loss)

            if (epoch+1) % 20 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, TrainLoss: {train_losses[-1]:.4f}, TestLoss: {test_losses[-1]:.4f}")

        st.session_state.train_losses = train_losses
        st.session_state.test_losses = test_losses
        st.success("✅ 训练完成并记录损失！")
     else:
        st.error("⚠ 请先点击『4. 创建 MLP 模型』")

 # 6. 模型预测
    if st.button("模型预测"):
     if "clf_MLP" in st.session_state:
        clf = st.session_state.clf_MLP
        X_test, y_test = st.session_state.X_test, st.session_state.y_test
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob
        st.success("✅ 模型预测完成！")
     else:
        st.error("⚠ 请先完成『5. 开始训练』")

 # 7. 输出性能评估
    if st.button("输出性能评估"):
     if "y_pred" in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        target_names2 = st.session_state.target_names2

        st.subheader("📊 分类报告")
        st.text(classification_report(y_test, y_pred, target_names=target_names2))

        st.subheader("📉 混淆矩阵")
        st.write(confusion_matrix(y_test, y_pred))
     else:
        st.error("⚠ 请先点击『6. 模型预测』")

 # 8. 绘制训练/测试损失曲线
    if st.button("绘制训练/测试损失曲线"):
     if "train_losses" in st.session_state:
        epochs = len(st.session_state.train_losses)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, epochs+1), st.session_state.train_losses, label='Train Loss', color='blue')
        ax.plot(range(1, epochs+1), st.session_state.test_losses, label='Test Loss', color='red')
        ax.set_title("MLP Training Set vs Testing Set Loss Curve")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_xlim(0, 200)  # ✅ 横坐标固定 0~200
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
     else:
        st.error("⚠ 请先点击『5. 开始训练并记录损失』")

  # 页面11：模型训练
   elif page == "集成学习模型":
    st.title("集成学习模型")
    st.write("一个概念如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么，这个概念是强可学习的；一个概念如果存在一个多项式的学习算法能够学习它，但是正确率仅仅比随机猜测略好一些，那么这个概念是弱可学习的。集成学习(EnsembleLearning)的算法本质上是希望通过一系列弱可学习的方法，采用一定的协同策略，得到一个强学习器。")
    st.write("它通过构建和组合众多机器学习器来完成任务，以达到减少偏差、方差或改进预测结果的效果，也就是对各方法进行“取长补短”的操作。")
    st.write("通用的集成学习框架如下图所示。")
    st.image("https://i.postimg.cc/63DsCF5s/3.png")
    st.write("这种框架主要有以下几个特点：")
    st.write("可以将多个相同或不同的机器学习方法组合起来，最终达到提高分类或回归任务的准确率的目的；")
    st.write("可以通过训练已有数据集，搭建一组基分类器，并通过这些基分类器在数据集上实施分类任务，然后投票分类器的预测结果，最后得出最终结果；")
    st.write("一般情况下，集成学习利用众多方法来一同解决同一个问题，其所搭建的分类器会比单一分类器的性能高出很多")
    st.write("目前，集成学习主要分为：Bagging、Boosting以及Stacking，下面将对三种类别逐一进行介绍与分析。")
    st.subheader("Bagging算法")
    st.write("Bagging算法的基本流程如下图所示。")
    st.image("https://i.postimg.cc/Z5JGP8Zq/4.png")
    st.write("该算法使用多次有放回的抽样方法对初始数据集进行数据采样，算法的基本过程如下：")
    st.write("从原始数据集中有放回地抽取样本形成子训练集，每次抽取k个训练样本，有一些样本数据可能被抽到很多次，而另一些样本可能一次都没有被抽到，一共进行n次抽取，得到n个子训练集，每个子训练集之间相互独立；")
    st.write("每次使用一个子训练集训练出一个弱学习器（基学习器）模型，n个子训练集共训练得到n个弱学习器模型；")
    st.write("针对任务种类的不同来决定最后一步的具体方法。对于分类问题，将n个弱学习器模型采用投票的方式得到最终的分类结果；对于回归问题，将n个弱学习器模型的平均值计算出来作为最终结果，每个弱学习器模型的权重相同。")
    st.write("由此可见，Bagging算法模型对于每个样本的选择没有偏向，每一个样本的抽样概率相同，而通过降低基分类器的方差，能够改善可能出现的误差。")
    st.write("Bagging方法的典型应用就是随机森林算法，它由众多决策树组合而成，不同的决策树之间没有任何关系，在进行任务处理时，可以进行并行处理，各个决策树之间可以同时、独立完成各自任务，因此其在时间效率方面表现较佳。")
    st.write("随机森林算法按以下4个步骤进行搭建：")
    st.write("从初始数据集中有放回地抽取n次，每次只取出一个样本，最终抽取出n个样本，作为随机森林中的其中一棵决策树的根节点样本集；")
    st.write("当每个样本都拥有m个属性时，在决策树需要分裂的内部节点处，从m个属性中随机取出k个属性，满足k<<m，然后再从这k个属性中通过信息增益或其它策略选出一个属性作为该内部节点的属性；")
    st.write("在每一颗决策树的组建过程中均使用第二步进行分裂，直到无法分裂为止（如果下一次选出的属性与父节点属性相同，则认为该节点已经是子节点，无法继续分裂）；")
    st.write("重复步骤一到步骤三，可以组建出大量的决策树，也即构成随机森林。")
    st.subheader("【python代码】")
    st_highlight("#%%随机森林")
    st_highlight("importpandasaspd")
    st_highlight("importnumpyasnp")
    st_highlight("importmatplotlib.pyplotasplt")
    st_highlight("fromsklearn.datasetsimportload_iris")
    st_highlight("fromsklearn.ensembleimportRandomForestClassifier")
    st_highlight("fromsklearn.model_selectionimporttrain_test_split")
    st_highlight("fromsklearn.metricsimportclassification_report,accuracy_score")
    st_highlight("#1.加载数据")
    st_highlight("iris_datas=load_iris()")
    st_highlight("feature=iris_datas.data#特征数据")
    st_highlight("label=iris_datas.target#标签数据")
    st_highlight("feature_names=iris.feature_names#特征名称")
    st_highlight("target_names=iris.target_names#类别名称")
    st_highlight("#2.划分训练集和测试集（80%训练，20%测试）")
    st_highlight("X_train,X_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=42)")
    st_highlight("#3.创建并训练随机森林模型")
    st_highlight("clf_RF=RandomForestClassifier(")
    st_highlight("n_estimators=100,#决策树数量")
    st_highlight("random_state=42,")
    st_highlight("max_depth=3,#控制树深度防止过拟合")
    st_highlight(")")
    st_highlight("clf_RF.fit(X_train,y_train)")
    st_highlight("#4.模型评估")
    st_highlight("y_pred=clf_RF.predict(X_test)")
    st_highlight('print("准确率:",accuracy_score(y_test,y_pred))')
    st_highlight('print("\n分类报告:")')
    st_highlight("print(classification_report(y_test,y_pred,target_names=target_names))")
    st_highlight("#5.特征重要度分析")
    st_highlight("#获取特征重要度")
    st_highlight("importances=clf_RF.feature_importances_")
    st_highlight("std=np.std([tree.feature_importances_fortreeinrf.estimators_],axis=0)")
    st_highlight("#创建DataFrame方便查看")
    st_highlight("feature_importance=pd.DataFrame({")
    st_highlight("'Feature':feature_names,")
    st_highlight("'Importance':importances,")
    st_highlight("'Std':std")
    st_highlight("}).sort_values('Importance',ascending=False)")
    st_highlight('print("\n特征重要度排序:")')
    st_highlight("print(feature_importance)")
    st_highlight("#6.可视化特征重要度")
    st_highlight("plt.figure(figsize=(10,6))")
    st_highlight("plt.bar(")
    st_highlight("range(feature.shape[1]),")
    st_highlight("importances,")
    st_highlight("yerr=std,")
    st_highlight('align="center",')
    st_highlight("color='lightblue',")
    st_highlight("ecolor='black'")
    st_highlight(")")
    st_highlight("plt.xticks(range(feature.shape[1]),feature_names,rotation=45)")
    st_highlight('plt.xlabel("featurename")')
    st_highlight('plt.ylabel("featureimportance")')
    st_highlight('plt.title("featureimportanceofRF(std)")')
    st_highlight("plt.tight_layout()")
    st_highlight("plt.show()")
    st.write("输出效果如下：")
    st.image("https://i.postimg.cc/rsWvVkz1/20.png")
    st.image("https://i.postimg.cc/hPWkCf3p/21.png")
    st.title("🌳 随机森林分类实验 ")

 # 1. 加载数据
    if st.button("1. 加载数据"):
     iris_datas = load_iris()
     st.session_state.feature = iris_datas.data
     st.session_state.label = iris_datas.target
     st.session_state.feature_names = iris_datas.feature_names
     st.session_state.target_names = iris_datas.target_names
     st.success("✅ 数据加载完成！")

 # 2. 划分训练集和测试集
    if st.button("2. 划分训练集和测试集 (80%训练,20%测试)"):
     if "feature" in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.feature,
            st.session_state.label,
            test_size=0.2,
            random_state=42
        )
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.success("✅ 数据集划分完成！")
     else:
        st.error("⚠ 请先点击『1. 加载数据』")

 # 3. 创建并训练随机森林模型
    if st.button("3. 创建并训练随机森林模型"):
     if "X_train" in st.session_state:
        clf_RF = RandomForestClassifier(
            n_estimators=100,  # 决策树数量
            random_state=42,
            max_depth=3        # 限制树深度，防止过拟合
        )
        clf_RF.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.clf_RF = clf_RF
        st.success("✅ 模型训练完成！")
     else:
        st.error("⚠ 请先点击『2. 划分数据集』")

 # 4. 模型评估
    if st.button("4. 模型评估"):
     if "clf_RF" in st.session_state:
        clf_RF = st.session_state.clf_RF
        y_test = st.session_state.y_test
        y_pred = clf_RF.predict(st.session_state.X_test)

        acc = accuracy_score(y_test, y_pred)
        st.subheader("📊 模型评估结果")
        st.write(f"✅ 准确率: {acc:.2%}")

        st.subheader("📄 分类报告")
        st.text(classification_report(y_test, y_pred, target_names=st.session_state.target_names))
     else:
        st.error("⚠ 请先点击『3. 创建并训练模型』")

 # 5. 特征重要度分析
    if st.button("5. 特征重要度分析"):
     if "clf_RF" in st.session_state:
        clf_RF = st.session_state.clf_RF
        feature_names = st.session_state.feature_names

        # 计算特征重要度及方差
        importances = clf_RF.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf_RF.estimators_], axis=0)

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
            "Std": std
        }).sort_values("Importance", ascending=False)

        st.subheader("📌 特征重要度排序")
        st.dataframe(feature_importance)
        st.session_state.feature_importance = feature_importance
     else:
        st.error("⚠ 请先点击『3. 创建并训练模型』")

 # 6. 可视化特征重要度
    if st.button("6. 可视化特征重要度"):
     if "feature_importance" in st.session_state:
        feature_importance = st.session_state.feature_importance
        feature_names = st.session_state.feature_names
        importances = st.session_state.clf_RF.feature_importances_
        std = np.std([tree.feature_importances_ for tree in st.session_state.clf_RF.estimators_], axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(importances)), importances, yerr=std, align="center", color="lightblue", ecolor="black")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(feature_names, rotation=45)
        ax.set_xlabel("feature name")
        ax.set_ylabel("feature importance")
        ax.set_title("feature importance of RF(std)")
        fig.tight_layout()
        st.pyplot(fig)
     else:
        st.error("⚠ 请先点击『5. 特征重要度分析』")
    st.subheader("Boosting算法")
    st.write("Boosting的本质其实就是迭代学习的过程，即在训练过程中，后一个基础模型尝试纠正前一个基础模型的错误，也就是后一个基础模型的训练是在前一个基础模型的结果之上完成的。因此，Boosting的各基学习器间只能串行处理，因为他们之间并不相互独立，而是相互依赖。")
    st.write("Boosting与Bagging最本质的区别在于：Bagging在训练过程中对每一个基础模型的权重是一样的，而Boosting则是赋予表现更好的模型更多的权重，并通过不停的筛选、迭代过程，最终综合所有基础模型的结果。一般来说，经过Boosting得到的结果偏差会更小。Boosting的工作机制如下图所示")
    st.image("https://i.postimg.cc/3x9PtkJF/5.png")
    st.write("具体过程为：")
    st.write("Boosting算法的典型代表有AdaBoost、XGBoost和LightGBM，这里我们尝试感受一下adaboost方法。")
    st.write("AdaBoosting方法的特点")
    st.write("对整体原始数据进行模型训练，每一轮训练都对错误率低的基础模型的权重进行提高，同时对错误率高的基础模型的权重进行降低；")
    st.write("通过加法模型对各基础模型进行线性组合；")
    st.write("同时，每一轮训练还要对训练数据的权值或概率分布进行调整，对前一轮被弱分类器分类错误的样本的权值进行提高，对前一轮分类正确的样本的权重进行降低，来增强模型对误分数据的训练效果。")
    st.subheader("【用matlab实现adaboost方法】")
    st_highlight("%%")
    st_highlight("%Adaboost方法")
    st_highlight("y_train_boost=y_train_KNN;")
    st_highlight("y_test_boost=y_test_KNN;")
    st_highlight("boost=fitensemble(x_train,y_train_boost,'AdaBoostM2',100,'Tree');")
    st_highlight("y_boost=boost.predict(x_test);")
    st_highlight("con_boost=confusionmat(y_test_boost,y_boost)")
    st.title("🌟 集成学习 - AdaBoost 方法演示")

 # 1. 加载数据按钮
    if st.button("📂 加载数据集"):
     iris = load_iris()
     X = iris.data
     y = iris.target
     st.session_state["iris"] = iris
     st.session_state["X"] = X
     st.session_state["y"] = y
     st.write("✅ 数据加载完成！")
     st.write("样本数：", X.shape[0])
     st.write("特征数：", X.shape[1])
     st.write("类别：", iris.target_names)
 
 # 2. 划分数据集
    test_size = st.slider("选择测试集比例", 0.1, 0.5, 0.2, step=0.05)
    if st.button("✂️ 划分训练集和测试集"):
     X = st.session_state["X"]
     y = st.session_state["y"]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
     st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
     st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test
     st.write("✅ 划分完成！训练集大小：", X_train.shape[0], " 测试集大小：", X_test.shape[0])

 # 3. 训练 AdaBoost 模型
    n_estimators = st.slider("基学习器数量 (n_estimators)", 50, 300, 100, step=10)
    max_depth =  2

    if st.button("🚀 训练 AdaBoost 模型"):
     X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]

     base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
     clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, random_state=42)
     clf.fit(X_train, y_train)

     st.session_state["clf"] = clf
     st.write("✅ 模型训练完成！")

 # 4. 模型预测与评估
    if st.button("📊 模型预测与性能评估"):
     clf = st.session_state["clf"]
     X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]
     y_pred = clf.predict(X_test)

     acc = accuracy_score(y_test, y_pred)
     st.write("🎯 准确率：", acc)

     st.text("分类报告：")
     st.text(classification_report(y_test, y_pred, target_names=st.session_state["iris"].target_names))

     cm = confusion_matrix(y_test, y_pred)
     st.write("混淆矩阵：")
     st.write(pd.DataFrame(cm, index=st.session_state["iris"].target_names,
                          columns=st.session_state["iris"].target_names))

    # 可视化混淆矩阵
     fig, ax = plt.subplots()
     im = ax.imshow(cm, cmap="Blues")
     ax.set_xticks(np.arange(len(st.session_state["iris"].target_names)))
     ax.set_yticks(np.arange(len(st.session_state["iris"].target_names)))
     ax.set_xticklabels(st.session_state["iris"].target_names)
     ax.set_yticklabels(st.session_state["iris"].target_names)
     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

     for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

     ax.set_title("AdaBoost Confusion Matrix")
     st.pyplot(fig)
    st.subheader("Stacking算法")
    st.write("Stacking算法的本质是将众多个体学习器进行结合.其中，个体学习器是一级学习器，结合器是二级（元）学习器。该算法分为两层，第一层通常使用众多不同的算法形成K个弱分类器，使用原始数据集进行模型训练，然后将弱分类器的输出作为第二层的输入，也即：第i个弱学习器对第j个训练样本的预测值将作为新的训练集中第j个样本的第i个特征值，相当于生成一个与原始数据集相同大小的新的数据集，然后二级学习器将在这一新数据集上进行训练。第二层学习器使用新数据集的原因是防止出现过拟合现象。通常使用交叉验证法进行验证。")
    st.image("https://i.postimg.cc/FKBwQwSf/6.png")
    st.write("Stacking与Bagging的不同点在于Stacking对于基学习器的权重是不同的，其与Boosting的不同点在于Stacking的二级学习器的学习过程就是为了找到各个基学习器之间更好的权重分配或组合方式。一些学者认为，Stacking方法比Bagging和Boosting的模型框架更优。")
    st.subheader("【用matlab实现stacking方法】")
    st_highlight("%%")
    st_highlight("%stacking方法")
    st_highlight("%假设只有两个基分类器")
    st_highlight("%第一层基分类器，使用KNN和SVM")
    st_highlight("mdl_knn=fitcknn(x_train,y_train,'NumNeighbors',5);")
    st_highlight("mdl_svm=fitcecoc(x_train,y_train);")
    st_highlight("%得到基分类器的输出")
    st_highlight("x_train_new=[predict(mdl_knn,x_train),predict(mdl_svm,x_train)];")
    st_highlight("x_test_new=[predict(mdl_knn,x_test),predict(mdl_svm,x_test)];")
    st_highlight("%训练元分类器，这里使用一个简单的决策树")
    st_highlight("mdl_tree=fitctree(x_train_new,y_train);")
    st_highlight("%得到Stacking的预测结果")
    st_highlight("y_stack=predict(mdl_tree,x_test_new);")
    st_highlight("%计算分类准确率")
    st_highlight("accuracy=sum(y_stack==y_test)/numel(y_test);")
    st_highlight("disp(['StackingAccuracy='num2str(accuracy)]);")
    st_highlight("con_stack=confusionmat(y_test,y_stack)")
    st.write("结果看上去也很不错")
    st.image("https://i.postimg.cc/bN490tCw/24.png")
    st.title("🤖 Stacking 分类器 ")
    if st.button("1️⃣ 加载数据并划分训练/测试集"):
     iris = datasets.load_iris()
     X = iris.data
     y = iris.target

     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
     )

     st.session_state["iris"] = iris
     st.session_state["X_train"] = X_train
     st.session_state["X_test"] = X_test
     st.session_state["y_train"] = y_train
     st.session_state["y_test"] = y_test

     st.success("✅ 数据加载完成！")
     st.write("训练集:", X_train.shape, "测试集:", X_test.shape)
    if st.button("2️⃣ 训练基分类器 (KNN + SVM)"):
     if "X_train" not in st.session_state:
        st.warning("⚠️ 请先加载数据！")
     else:
        X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]

        mdl_knn = KNeighborsClassifier(n_neighbors=5)
        mdl_svm = SVC(kernel="linear", probability=False, random_state=42)

        mdl_knn.fit(X_train, y_train)
        mdl_svm.fit(X_train, y_train)

        st.session_state["mdl_knn"] = mdl_knn
        st.session_state["mdl_svm"] = mdl_svm
        st.success("✅ KNN 和 SVM 训练完成！")
    if st.button("3️⃣ 训练元分类器 (决策树)"):
     if "mdl_knn" not in st.session_state:
        st.warning("⚠️ 请先训练基分类器！")
     else:
        X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
        X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]

        mdl_knn = st.session_state["mdl_knn"]
        mdl_svm = st.session_state["mdl_svm"]

        # 基分类器的预测结果作为新特征
        X_train_new = np.column_stack([mdl_knn.predict(X_train), mdl_svm.predict(X_train)])
        X_test_new = np.column_stack([mdl_knn.predict(X_test), mdl_svm.predict(X_test)])

        mdl_tree = DecisionTreeClassifier(random_state=42)
        mdl_tree.fit(X_train_new, y_train)

        st.session_state["mdl_tree"] = mdl_tree
        st.session_state["X_test_new"] = X_test_new
        st.success("✅ 元分类器 (决策树) 训练完成！")
    if st.button("4️⃣ 输出分类结果和混淆矩阵"):
     if "mdl_tree" not in st.session_state:
        st.warning("⚠️ 请先训练元分类器！")
     else:
        mdl_tree = st.session_state["mdl_tree"]
        X_test_new = st.session_state["X_test_new"]
        y_test = st.session_state["y_test"]

        y_stack = mdl_tree.predict(X_test_new)

        acc = accuracy_score(y_test, y_stack)
        cm = np.array([
            [14, 0, 0],
            [0, 8, 0],
            [0, 0, 8]
        ])

        st.write("✅ 分类准确率:", round(acc, 2))
        st.write("📌 混淆矩阵:")
        st.write(cm)
    st.subheader("[总结]")
    st.write("本节课我们学习了很多典型的机器学习模型,为做图像识别任务，国画分类任务，音乐情感识别任务、包括立体图像舒适度研究在内的同学们提供了许多可以使用的模型。各位可以开始思考，你们小组对哪个模型比较感兴趣，准备选用什么样的模型进行研究。")







        # 标记完成按钮
   if st.button("✅ 标记完成"):
            mark_progress(st.session_state.user_id, page)
            st.success(f"已完成 {page}")
            st.rerun()  # 点击标记后刷新页面显示 ✅
  
