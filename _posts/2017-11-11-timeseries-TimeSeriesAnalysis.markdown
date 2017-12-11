---
layout: post
title:  "时间序列分析"
date:   2017-11-11 11:31:28 +0800
categories: [timeseries]
---

* TOC
{:toc}

## 业务背景
品牌广告通常采用CPM业务模式，特点在于保价保量，需要我们准确预估各个广告位在未来一段时间的可用库存，以便合理售卖广告资源。通常，品牌广告提前14-28天下单占比接近60%，因此降低短期预估误差可以更好指导售卖；另外，游戏等行业客户普遍存在提前3个月下单，降低长期预估误差可以更好服务细分需求。

通过分析腾讯视频在PC端、移动端、TV端最近三年的的库存数据，我们发现库存不仅在每周表现出$weekly$波动性（例如，双休日高于工作日），另外在每个月也表现出一定的$yearly$波动（例如，暑期比开学高，除夕前后比较低）。之前大部分业务有效数据积累不足两年，不能有效探索数据长期规律。因而难以捕捉到库存的季节变化，造成较高的预估偏差（例如暑期结束开学，如果没有考虑到开学季库存大幅下降的特性，会造成预估过高的现象；又如国庆之前预估的训练集为下降趋势，会造成国庆节预估过低的现象），往往需要人工干预来调整预估库存。

近来，我们通过综合分析多个维度的长期数据，挖掘其中的长周期（全年各个月份的波动差异）以及阶段周期（暑假的$weekly$周期与非暑假有差异）等规律，从而进一步降低品牌广告的预估误差，取得了一定的效果，具体如下：
1. 提前14天的基线算法预估误差为10.2%，新算法预估误差为8.9%，预估误差同期下降12.7%；
2.  提前120天的基线算法预估误差为17.4%，新算法预估误差为13.7%，预估误差同期下降21.3%；

通过对库存时间序列预估的分析，我们在多周期时间序列预估方面有了部分积累，对时间序列的平稳、趋势、周期、自回归等规律有了更多认识。在调研过程中，我们观察到在天气、电商、金融等各行各业都有挖掘时间序列的特性，预估未来走势，从而更好指导生产、分配以及投资等需求。“虽世殊事异，所以兴怀，其致一也”，我们通过总结梳理多方面文献与资料，试图对时间序列分析做一个简要的介绍，希望能够对相关场景预估有一定的帮助。

## 算法简介

到了19世纪，概率论的发展从对（相对静态的）随机变量的研究发展到对随机变量的时间序列$s_1,s_2,s_3....s_t,...$,即随机过程（动态的）研究。在哲学的意义上，这是人类认识的一个飞跃。但是，随机过程要比随机变量复杂得多。首先，在任何一个时刻$t$，对应的状态$s_t$都是随机的。时间序列分析($Time\ Series\ Aanlysis,TSA$)是一种动态数据处理的统计方法。该方法基于随机过程理论和数理统计学方法，研究随机数据序列所遵从的统计规律用于解决实际问题。业界很多公司，都结合自己业务需求，分析时间序列的特性，借此来达到预测或者异常点检测的目的。

 2017年2月份，$Facebook$宣布开源了一款基于$Python$和$R$语言的时间序列预测工具---[$Prophet$](https://facebook.github.io/prophet/)，即"先知"。作者$Benjamin\ Letham$在对应的论文《$Forecasting\ at\ Scale$》中提到，提出$Prophet$的目的在于希望帮助大量的分析师在众多场景下解决大数据量下的时间预测问题，这就是"$at\ Scale$"的内涵。

全球最大最优质的搜索引擎$Google$针对搜索量、收入等业务指标也有预估需求，在实际业务中采用多种模型集成提升预估精度。并且开源了基于贝叶斯结构化时间序列分析方法[$bsts(Bayesian\ Structual\ Time\ Series)$](https://www.rdocumentation.org/packages/bsts/versions/0.7.1/topics/bsts)。由于现实数据大部分是非平稳的，$Vitaly\ Kuznetsov$提出来基于差异量的[$“DBP”(Discrepancy\ based\ Forecast)$](https://cs.nyu.edu/~mohri/talks/NIPSTutorial2016.pdf)来预估非平稳时间序列序列也取得了良好的效果。

在时间序列中，异常检测也是一个重要问题。$Twitter$号称"地球脉搏",流量异常检测对其运营有着不小的挑战，其在2015年开源了一个$R$语言的算法包[$Anomaly\ Detection$](https://github.com/twitter/AnomalyDetection)借此，$Twitter$通常会在重大新闻和体育赛事期间扫描入站流量，发现那些使用僵尸账号发送大量垃圾（营销）信息的机器人。

$Netflix$作为世界上最大的在线影片租赁服务商，拥有大量的服务器集群以支撑在线影片的存储以及个性化影片推荐等服务。其在2014左右开源了[$Robust\ Anomaly\ Detection$](https://github.com/Netflix/Surus),借此，能自动发现“不健康”的服务器,快速修复问题，减轻运维人员的负担。

我们最近对品牌广告的库存预估进行分析时，发现结合长周期可以有效捕捉时间序列的长周期变化规律，从而进一步提升库存预估精度。时间序列是一个通用的问题，广泛存在于通信、气象、金融、交通、销售等各行各业。由于时间序列分析与数字信号处理$(DSP,Digital\ Signal\ Processing)$有着很强的关联系，分析方法也有很多共同点。本文将主要针对预估，结合两者从时间序列的分解特性、预测原理、使用方法以及业界动态略作介绍。

需要注意的是，时间序列分析通常看成是统计学家的工作。最常用的是$R$语言，其次是$Python$。如果能够掌握$R$语言，则拥有了大量的分析工具，可以快速验证分析方法以及预估策略的有效性。本文在结合品牌广告库存的分析基础上，参考相关博文、教材以及论文等，统一其变量等，梳理得到本文。

## 主要内容
介绍时间序列分析的基础知识，如平稳性、相关性、周期性等；
介绍线性时间序列分析的原理，如$AR、MA、ARMA、ARIMA$等；
介绍非线性时间序列分析的原理，如$RNN、LSTM$等；
介绍在时间序列分析在$Python$和$R$中的应用方法；
介绍业界在时间序列分析方面的探索，如$Facebook$、$Google$等；

##   基础知识
### 时间序列分解
$Rob \ J\  Hyndman$在[《Forecasting: principles and practice》](https://www.otexts.org/fpp/6/1)指出，一个时间序列通常由4种成分构成：分别是趋势($Trend,T_t$)、季节变动($Seasonal,S_t$)、循环变动($Circular,C_t$)和不规则变动($Irregular,I_t$)。时间序列$Y_t$可以表示为以上4个因素的函数，即：
$$X_t=f(T_t,S_t,C_t,I_t)  \tag{1}$$
时间序列分解的方法有很多，较常用的模型有加法模型和乘法模型。
加法模型为：
$$X_t=T_t+S_t+C_t+I_t \tag{2}$$
乘法模型为：
$$X_t=T_t\times S_t\times C_t\times I_t \tag{3}$$
加法模型如某品牌服饰在冬季的销量提升100万，乘法模型则如该服饰在冬季的销量增长$20\%$。一般而言，乘法模型可以通过取对数转化成加法模型；季节变动与循环变动存在差异；趋势可能会存在某个$change \ point$由增长转变为下降；去掉趋势和周期项之后的序列可能存在自回归特性。下面举一个简单的例子：
假设我们需要根据2017年7月1日之前的历史数据来预估未来4个月的库存情况，下图是2017年7月1日之前的历史库存(注：纵坐标为幅度，已经作了脱敏处理)：
![某个维度历史库存](/img/history.png)
结合上图的数据和一些时间序列的分析方法（后文将简要介绍），可以大致看出以下先验认识：
- 最近三年的该维度视频库存总量在稳步增长
- 在每年的除夕附近有大幅的下降（例如上图中2015年2月份的局部极小点实际为除夕当天）
- 在每年各个月份之间会有不同幅度的震荡（例如2016年8月份暑假明显高于2016年9月开学）
- 在每周中表现出一定的波动（例如图中的众多小“毛刺”点）
- 在暑假的波动幅度相对非暑假要小（例如2016年8月份暑假“毛刺”波动幅度比2016年9月份明显要小）

如果希望通过传统的时间序列分析方法来预估未来，尤其是未来四个月的每日库存，则需要我们充分挖掘历史库存的信息量，这里我们以改进后的贝叶斯结构化模型$Prophet$为例来分析历史库存的重要构成成分。通过结合我们对库存数据的先验认识以及假设的似然函数，通过最大后验估计可以分解历史库存数据如下图：
分解历史数据得到下图：
![时间序列分解](/img/with_summer_dpi_nofuture.png)

可以看到如下规律：
- 从trend趋势看，最近三年数据趋势稳步增长，在部分$change\ point$增长速率有变化；
- 从holiday节日看，每年除夕相对前后暴跌近30%，9月开学会下降，国庆节又会比较高；
- 从yearly周期来看，每年暑假（7月-8月）相对开学后（9月-10月）的波动高达20%左右；
- 从非暑假weekly周期看，每周六日的库存量相比工作日的波动在15%左右；
- 从暑假weekly周期看，每周六日的库存量相比工作日的波动在8%左右，比非暑假要小；

如果能够把握住上面所提到五个重要构成的规律，我们可以进一步预估未来4个月的库存量，如下图：
![预估未来四个月库存](/img/compare.png)
上图中，横坐标表示时间，纵坐标表示取对数后并且归一化的某个维度库存数据，蓝色表示真实库存，绿色表示未来四个月的预估值，上图中未来120天的预估误差均值不超过8%，能有效克服把握住历史的长期增长趋势、全年季节的波动、暑假中的每周较小幅度波动以及非暑假每周较大幅度波动。可以看出，如果充分考虑到库存的季节变化规律，提取得到各个成分，将有利于降低库存预估误差，并且有助于用来预估更长时间跨度。

然而，实际的业务数据存在各式各样的特点，例如历史数据不够长、波动性大、趋势由增长转下降、考察序列的粒度有分钟级或者秒级差异等，直接套用开源工具包不一定会取得良好的效果。此时需要回归到传统时间序列分析的道路上来，具体分析当前业务某个维度数据的平稳性、周期性、趋势、自回归特性等，可能需要分析时间序列的时间频率特性来验证直观的认识，也可能需要引入例如$LSTM$这样的模型来预估，本文将对这些基础性的工具和方法略作介绍。

### 平稳性
>引言：平稳性是时间序列分析的基础。然而实际数据大多是非平稳的，例如服务器在某个时段任务密集，负载存在波动性；从长期看，负载可能逐渐上升，体现出非平稳性。强平稳条件比较严苛，一般难以满足；一般通过时间序列的分解，去掉趋势和周期项，得到弱平稳序列，从而研究其自回归特性。平稳性的定义和划分可以参考《概率论与数理统计-盛骤》与《金融时间序列分析-蔡瑞胸》。

- 强平稳
时间序列${r_t}$陈为严平稳的($Strictly\ Stationary$)，如果对所有的$t$，任意正整数$k$和任意$k$个正整数($t_1,t_2,...,t_k$),($r_{t_1},r_{t_2},...,r_{t_k}$)的联合分布与($r_{t_1+t},r_{t_2+t},...,r_{t_k+t}$)的联合分布是相同的。即严平稳性要求($r_{t_1},r_{t_2},...,r_{t_k}$)的联合分布在时间的平移变换下保持不变（注：要求联合分布保持不变，并非要求方差不变）。这是一个很强的条件。通常的时间序列并不满足，并且难以用经验方法验证。经常假定的是平稳性的一个较弱的形式。
- 弱平稳
如果$\forall l\in \mathcal{Z}$,$r_t$的均值以及$r_t$和$r_{t-l}$的协方差均不随时间而改变。即有满足如下两个条件：
$$
\begin{equation}  
\left\{  
             \begin{array}{lr}  
             E(r_t)=\mu, \\
             Cov(r_t,r_{t-l})=r_l  &
      \end{array}
      \tag{4}
\right.  
\end{equation}  
 $$
其中$\mu$是一个常数，$r_l$只依赖于$l$。
弱平稳性意味着数据的时间图显示出$T$个值在一个常数水平上下以相同的幅度波动。在应用中，弱平稳性使我们可以对未来观测进行推断，即预测。

### 相关性
>引言：序列潜在的相关性是预测的前提，如果时间序列之间相互独立，则难以进行有效预估。在衡量相关性方面，主要有自相关和偏自相关。需要关注的是二者的区别，偏自相关衡量的是过去单项对当前的影响程度。偏自相关的定义可以参考《金融时间序列分析-蔡瑞胸》和《时间序列分析-汉密尔顿》。

事件序列的自相关性一般可以用时间序列的如下三个统计量来体现：
- 自协方差函数($Autocovariance\ Function$)
- 自相关系数函数($Autocorrelation\ Coefficient\ Function,ACF$)
- 偏自相关系数函数($Partial\ Autocorrelation\ Coefficient\ Function,PACF$)

具体如下：
- ACF
对于单个随机变量$X$，可以计算其均值$\mu$，方差$\sigma^2$；
对于两个随机变量$X和Y$，可以计算$X和Y$的协方差：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
      cov(X,Y)=E[(X-\mu_X)(Y-\mu_Y)] & \\
      \rho(X,Y)=\dfrac{cov(X,Y)}{\sigma_X\sigma_Y}
      \end{array}  
      \tag{5}
\right.  
\end{equation}  
$$
他们衡量了两个不同事件之间的相互影响程度。
而对于时间序列${X_t,t \in T}$，任意时刻的序列值$X_t$都是一个随机变量，每一个随机变量都会有均值和方差，记$X_t$的均值为$\mu_t$和方差$\sigma_t$。则$\forall t,s \in T$，定义$X_t$的自协方差函数$\gamma(t,s)$和自协方差系数$\rho(t,s)$如下：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
     \gamma(t,s)=E[(X_t-\mu_t)(X_s-\mu_s)] & \\
    \rho(t,s)=\dfrac{cov(X_t,X_s)}{\sigma_t\sigma_s}
    \end{array}  
    \tag{6}
\right.  
\end{equation}  
$$
$\gamma(t,s)$和$\rho(t,s)$衡量的是同一个事件在两个不同时期（时刻$t$和$s$）之间的相关程度，形象地讲就是衡量自己过去的行为对自己现在的影响。

- PACF
假设$X_t$的一阶自相关系数$\rho_1 > 0$,则表明$X_t$与$X_{t-1}$相关，不过$X_{t-1}$又与$X_{t-2}$相关，$X_{t-3}$也会受到$X_{t-4}$的影响$...$。因此当我们计算$X_{t}$与$X_{t-1}$的自相关系数时，实际上我们既捕捉到了$t-1$项信息对$t$的影响，捕捉到了${t-2}$项信息对$t$的影响$...$。总而言之，$X_t$与$X_{t-1}$的自相关系数实际衡量的是包括过去数据对今天的影响总和。为了衡量过去单项对现在的影响效果，剔除其他项的作用，因此引入了偏自相关系数PACF。
假设$X_t$对$X_{t-1}、X_{t-2}、...X_{t-k}$的$k$阶线性回归方程：
$$X_t=\phi_{k,1}X_{t-1}+\phi_{k,2}X_{t-2}+...+\phi_{k,k}X_{t-k}+\epsilon_{k,t}\ \tag{7}$$
其中所有参数$\phi_{k,j}(j=1,2,...,k)$的估计值应确保残差$\epsilon_{k,t}$的方差
$$\delta_k=E[X_t-\sum_{j=1}^{k}\phi_{k,j}X_{t-j}]^2=\gamma_0-2\sum_{j=1}^k\phi_{k,j}\gamma_j+\sum_{i,j=1}^{k}\phi_{k,i}\phi_{k,j}\gamma_{i-j} \tag{8}$$
达到最小。
为此，类似于经典回归模型参数估计的最小二乘法，将$\delta_k$分别关于$\phi_{k,j}(j=1,2,...,k)$求偏导数，并令相应的结果为0，于是得到如下形式的线性方程组：
$$\gamma_j=\sum_{i=1}^{k}\phi_{k,i}\gamma_{i-j},(j=1,2,...,k) \tag{9}$$
展开可以得到如下：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
     \gamma_1=\phi_{k,1}\gamma_0+\phi_{k,2}\gamma_1+...+\phi_{k,k}\gamma_{k-1} & \\
      \gamma_2=\phi_{k,1}\gamma1+\phi_{k,2}\gamma_0+...+\phi_{k,k}\gamma_{k-2} & \\
      ... & \\
       \gamma_k=\phi_{k,1}\gamma_{k-1}+\phi_{k,2}\gamma_{k-2} +...+\phi_{k,k}\gamma_0
   \end{array}  
   \tag{10}
\right.  
\end{equation}  
$$
考虑到$\gamma_0=1$,上述方程可以进一步转化为：
$$
\begin{gather*}
\begin{bmatrix} r_1 \\ r_2 \\ . \\ .\\r_k \end{bmatrix}\quad=
\begin{bmatrix} 1 & r_1 &r_2 & ... & r_{k-1} \\1 & r_1 &r_2 & ... & r_{k-1} \\.  & . & . & ... & . \\.  & . & . & ... & . \\1 & r_1 &r_2 & ... & r_{k-1}    \end{bmatrix} \quad
\begin{bmatrix} \phi_1 \\ \phi_2 \\ . \\ . \\ \phi_k \end{bmatrix}\quad
\tag{11}
\end{gather*}
$$
上式称之为[$Yule-Walker$](http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YWSourceFiles/YW-Eshel.pdf)方程。从中可求解出$\phi_{k,1}、\phi_{k,2}、...、\phi_{k,k}$。其中的最后一个解$\phi_{k,k}$便为时间序列$X_t$延迟$k$阶偏自相关系数$PACF$。

### 周期性
>引言：傅里叶分析在物理学、信号学、密码学、声学以及结构动力学等诸多领域都有着广泛的应用，上个世纪40年代以来，Fourier变换伴随着FFT(快速傅里叶变换)算法的快速发展而成为极其重要的数学工具。傅里叶变换可以用来挖掘时间序列的潜在周期模式，帮助提高预估准确性。下面，将按照傅里叶级数、傅里叶变换到离散傅里叶变换的顺序稍作介绍。傅里叶分析可以参考《数学物理方法-胡学刚》、《数学分析-陈传璋》、《信号与系统-郑君里》、《数字信号处理-程培青》以及《通信原理-周炯槃》。

- 傅里叶级数
1822年，法国数学家傅里叶($J.Fourier,1768-1830$)在研究热传导理论时发表了“热点解析理论”，提出并证明了将周期函数展开为正弦级数的原理，奠定了傅里叶级数的理论基础。下面给出周期信号$\tilde{x}(t)$的傅里叶级数表达式：
$$\tilde{x}(t)=\sum_{k=-\infty}^{+\infty} a_k e^{jkw_0t}\tag{12} $$
其中$j$为虚数单位，$w_0$为信号$\tilde{x}(t)$的基频，即为$w_0=\dfrac{2\pi}{T}$,$a_k$的计算如下式：
$$a_k=\dfrac{1}{T}\int_{-T/2}^{T/2}\tilde{x}(t)e^{-jkw_0t}dt \tag{13}$$
周期的意义在于限定傅里叶系数的积分公式上下限,将$(13)$式带入到$(12)$中，两边恒等，其中利用了$e$指数的正交性。
- 连续傅里叶变换
然而这并不够，傅里叶级数只限定在周期信号，如果针对非周期信号$(12)$中的基频$w_0$将不存在（或者说无穷小，即$w_0\to0$），为了扩展应用范围，数学家们尝试定义非周期信号的傅里叶变换。区别在于非周期信号的基频$w_0\to0$，无穷小的求和可以用积分形式来表达。
假设上述周期信号$\tilde{x}(t)$就是根据$x(t)$扩展而得到，那么对于$\left| t  \right|\leq T/2$,有$x(t)$=$\tilde{x}(t)$。对于周期信号$\tilde{x}(t)$，对应的傅里叶系数为:
$$a_k=\lim_{T\to \infty}\dfrac{1}{T}\int_{-T/2}^{T/2}\tilde{x}(t)e^{-jkw_0t}dt \\
=\lim_{T\to \infty}\dfrac{1}{T}\int_{-T/2}^{T/2}x(t)e^{-jkw_0t}dt \tag{14}\\
=\lim_{T\to \infty}\dfrac{1}{T}\int_{-\infty}^{+\infty}x(t)e^{-jkw_0t}dt  
$$
现在定义$X(jw)$为$Ta_k$的包络，其中的$kw_0$用$w$来代替，则得到如下：
$$X(jw)=\int_{-\infty}^{+\infty}x(t)e^{-jwt}dt \tag{16}$$
显然，$a_k$只是$X(jw)$的等间隔采样：
$$a_k=\dfrac{1}{T}X(jkw_0) \tag{17}$$
另外这里还可以注意到：
$$\tilde{x}(t)=\sum_{k=-\infty}^{+\infty} a_k e^{jkw_0t} \tag{18} \\
=\sum_{k=-\infty}^{+\infty}\dfrac{1}{T}X(jkw_0)e^{jkw_0t} \\
=\sum_{k=-\infty}^{+\infty}\dfrac{1}{2\pi}X(jkw_0)e^{jkw_0t}w_0
$$
观察上式，发现$\tilde{x}(t)$只由$w_0$与$kw_0$构成，即时域周期信号可以表示为离散的频域信号的组合。这表明傅里叶变换具有“时域周期则频域离散”的性质。
- 离散时间傅里叶变换DTFT
所谓离散时间，可以理解为"序列"。离散时间傅里叶变换也称为序列的傅里叶变换。其表达式如下：
$$X(jw)=\sum_{n=-\infty}^{+\infty}x(n)e^{-jwn} \tag{19}$$
可以看出$X(j(w+2\pi M))=X(jw)$,即DTFT的频率是$w$的连续周期函数，周期为$2\pi$.
DTFT的时频特性可以总结为：时域离散，频率连续且以$2\pi$为周期。其中$0,2\pi,4\pi...$对应直流分量，$\pi,3\pi，5\pi...$对应信号的高频分量。
时域离散，频域周期。
- 离散傅里叶变换DFT
DFT在时域和频域都是离散的。可以看成是$Discrete\ time\ frequency \ Fourier\  Transform,DTFFT$的简称.

$$\begin{align}
&X(k)=\sum_{n=0}^{n=N}x(n)W_N^{nk},k=0,1,...,N-1  \tag{20} \newline
\ & x(n)=\dfrac{1}{N}\sum_{k=0}^{N-1}W_N^{-nk},n=0,1,...,N-1 \tag{21}
\end{align}$$
 其中$W_N^{nk}=e^{-j\dfrac{2\pi}{N}nk}$.DFT的两个特性：时域和频域都是离散的；时域和频域点数都是有限的。可以方便在计算机中进行计算实现。且时域和频域均可在一个周期内完全反映出来。
$FFT$是$DFT$的一种高效算法，称为快速傅里叶变换($fast\space fourier\space transform$)。
傅里叶变换在时间序列分析中有重要的应用，下面举两个例子：

（1）时间序列的多周期性
![时间序列的多周期性](/img/fft_long.jpg)

- 时域特性分析：
上图为某个维度最近三年归一化库存数据，其中的黄色圆圈表示每年除夕的库存暴降，紫色圆圈代表的是每年暑假的库存上升以及开学后库存暴降的过程。整个库存展现出增长的趋势，年份之间还表现出一定的波动性。
- 频域特性分析：
根据上述离散傅里叶DFT特性的分析，频率取值为：
$w_k=\dfrac{2\pi}{N}\tag{22}k$
或者：
$f_k=\dfrac{1}{N}k \tag{23}$
则对应得到信号的时域周期：
$n_k=\dfrac{1}{f_k}=\dfrac{N}{k} \tag{24}$
可以看到图中$T=365$、$T=7$以及$T=4$周期成分表现较为明显。
如果能够充分提取到库存序列的趋势、多周期特性则能有效提升预估性能。

（2）时间序列的阶段周期性

平时和暑假的周期特性表现有差异，暑假的周期中$T=7$表现不明显。
![暑假和非暑假的week周期波动差异](/img/fft_period1.jpg)
暑假的数据的周期成分低，波动性小；根据离散傅里叶变换的定义，用总共时间长度60除以横坐标，得到的是周期数，例如对红色线在17处，实际是60/17=3.5左右的周期

（3）低通滤波器过滤异常点

可以通过对时间序列的频谱进行低通滤波，过滤掉数据中异常存在的点。

### 模型识别
>引言：经典的时间序列分析方法，通常采用自回归滑动平均的方法来建模，需要选择最优的参数值，然后预测模型。通常有两种方法：1.采用前述的相关函数法，通过相关函数的变化来推测模型参数的取值；2.定义模型的选择指标，兼顾模型的表达能力与简单性。可以参考《金融时间序列分析-蔡瑞胸》。

#### 相关函数法
- ACF
- PACF
#### 信息准则法
$AIC$是一种用于模型选择的指标，同时考虑了模型的拟合程度以及简单性，而$BIC$则是对$AIC$的一个改进，具体的定义如下：
- AIC准则
$Akaike$信息准则($AIC$)定义如下：
$$AIC=\dfrac{-2}{T}ln(似然函数的最大值)+\dfrac{l\cdot 2}{T}\tag{25}$$
其中$T$是样本容量,$l$是参数的个数。第一项度量的是模型对数据的拟合优度，而第二项称为准则中的惩罚函数。参数个数越多，第二项的惩罚越大。
- BIC准则
$Schwarz$贝叶斯信息准则($BIC$)定义如下：
$$BIC=ln(\widetilde{\sigma}_l^2)+\dfrac{l\cdot ln(T)}{T}\tag{26}$$
在$AIC$准则中对每个参数的惩罚为2，而在$BIC$中为$ln(T)$，当样本容量适度或较大时，$BIC$会更倾向于一个低阶的模型。
一般而言，较小的$AIC$或者$BIC$表明模型在保持简单的同时能够很好地对时间序列进行拟合。因此，我们往往会选择具有最小的$AIC$或者$BIC$的模型作为相对最优的模型。

### 模型检验
>引言：时间序列的平稳性是许多时间序列分析方法的前提，在构建模型用来预测未来之前，需要检验该时间序列的平稳性。检验方法一般有三种：观察时序图、观察相关函数图以及单位根检验法。另外，应用模型提取时间序列的趋势、周期、自回归等特性后，还需要检测残差序列是否为白噪声，如果不是白噪声，从信息论角度看，模型还没有完全表达序列数据，即为欠拟合。

#### 平稳性检验
- 时序图观察法
  根据弱平稳的定义，序列的均值和方差是一个常数。也就是说平稳时间序列的序列值在一个常数水平上下波动，并且波动幅度接近。因此，如果序列数据围绕一个水平线上下以大致相同的幅度波动，那么该时间序列可能具备弱平稳性。
- 相关函数观察法
一般对平稳时间序列而言，其自相关函数或者偏自相关函数大都可以快速减小到0或者在某一个阶段之后变为0,而非平稳序列一般不具备这样的性质。
- 单位根检验
上述两种方法都是通过观察的方法来完成，存在一定的主观性。单位根检验可以比较客观判断序列的平稳特性。如果时间序列是非平稳的，可以通过$d$次差分($Difference$)可以将其转化为平稳序列，称差分得到的序列为$I(d)$，其中$I$意指整合($Integrated$),$d$为整合阶数，例如：
$$x_t=x_{t-1}+\epsilon_t\tag{27}$$
其中$x_0=0$,$\epsilon_t \sim N(0,\sigma^2_\epsilon)$,则可以得到：
$$E(x_t)=0 \tag{28}$$
$$Var(x_t)=Var(x_{t-1})+\sigma^2_\epsilon=...=t\sigma^2_\epsilon\tag{29}$$
因为$x_t$的方差会随着时间而改变，因此该时间序列是非平稳的。然而$x_t$经过一阶差分之后将为平稳的时间序列，因此$x_t$为非平稳的$I(1)$序列。一个时间序列是否平稳可以借助于滞后算子($Lag\ Operator$)多项式方程的根来表述。滞后算子是将一个时间序列的前一个值转化为当前值，通常用$L$表示，即$Lx_t=x_{t-1}$(有的领域文献或者以$B$来表示，取自$BackShift$,同样表示后移算子的意思)。在信号与系统中，也常引入$L$来分析系统的稳定性。那么上式可以改写为：
$$(1-L)x_t=\epsilon_t\tag{30}$$
其中$1-L=0$称为滞后算子多项式，得到根$L=1$。因此$x_t$称为单位根过程。时间序列${x_t}$是随机游走过程，其滞后算子多项式的根有单位根，是个非平稳的时间序列。如果序列非平稳，则预估难以进行，通常采用$DF(Dickey-Fuller\ Test)$检验或者$ADF(Augmented Dickey-Fuller\ Test)$检验来确认序列是否存在单位根。
假设一个简单的$AR(1)$模型是$y_t=\rho y_{t-1}+\epsilon_t$,如果$\left |\rho \right | \geq 1$，则说明单位根是存在的。回归模型可以进一步写成：
$$\Delta y_t=(\rho-1)y_{t-1}+\epsilon_t\tag{31}$$
其中$\Delta$是一阶差分。测试是否存在单位根等同于测试否$\rho=1$.$ADF$检验与$DF$检验类似，但ADF检验的好处在于它排除了自相关的影响。$ADF$检验的模型为：
$$\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\delta_1\Delta y_{t-1}+...++\delta_p\Delta y_{t-p}+\epsilon_t\tag{32}$$
其中$\alpha$为对应截距，$\beta$为对应趋势。针对ADF的假设检验为：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    H_0：\gamma=0 &\\
    H_1：\gamma=1
   \end{array}  
   \tag{32}
\right.  
\end{equation}  
$$
该检验对应的统计量为：
$$DF=\dfrac{\hat \gamma}{SE(\gamma )}\tag{33}$$
如果该统计量比临界值小，则拒绝原假设，也就是认为序列是平稳的，否则认为序列是非平稳的。

#### 白噪声检验
时间序列${r_t}$称为一个白噪声序列，如果${r_t}$是一个具有有限均值和有限方差的独立同分布随机变量序列。特别地，若${r_t}$还服从均值为0、方差为$\sigma^2$的正态分布，则称这个序列为高斯白噪声。对白噪声序列，所有自相关函数为0。在实际应用中，如果所有样本自相关函数接近于0，则认为该序列是白噪声序列。
白噪声是一种理想存在；信息熵最大，完全未知的情况；
白噪声检验也成为纯随机性检验，一般是构造检验统计量来检验序列的纯随机性，常见的检验统计量有$Q$统计量、$LB$统计量由样本各延迟期数的自相关系数可以计算得到检验统计量，然后计算得到对应的$p$值
，如果$p$值显著大于显著性水平$alpha$，则表示该序列不能拒绝纯随机的原假设，该序列已经没有什么可以挖掘的有用信息，因而可以停止对该序列的分析。
通常我们用$Ljung-Box$检验，简称$LB$检验。简单来说$LB$检验的原假设为所检验序列为纯随机序列(白噪声过程),该检验的统计量为$Q$统计量：
$$Q(m)=n(n+2)\sum_{1}^m\dfrac{\rho_k^2}{n-k}\sim\chi_m^2\tag{34}$$
其中，$\rho_k^2$是序列的$k$阶自相关系数，$n$是整个序列中观测值的个数，$m$是设定的滞后阶数。
根据$Q(m)$公式可知，$Q(m)$均为正数并且与数值大小与序列的自相关系数呈正相关。即当序列有自相关时，其自相关系数较大、对应的$Q(m)$也较大；反之，当序列为随机序列、无自相关性时，序列的自相关系数不会显著地异于0，$Q(m)$也会很小。
检验一个时间序列在$m$阶内是否是白噪声，只有当$Q(1),Q(2),...,Q(m)$这$m$个统计量都小于对应的$\chi^2$分布的临界值，才能说明该序列在所检验的$m$阶内是纯随机的。
>$LB$检验在$R$中可以调用$Box.test()函数来实现。$

### 参数估计
>引言：通过分析数据特性，然后假设序列服从某种先验的规律之后，例如指定逻辑回归形式的增长函数，接下来需要进一步确定模型的参数。通常由两种典型的方法：最大似然法和最大后验法。分别属于频率学派和贝叶斯学派。对于参数估计的理论可以参考《Pattern Recognition and Machine Learning- Christopher Bishop 》以及《The Elements of Statistical Learning-Trevor Hastie》。

最大似然法$MLE$与最大后验法$MAP$来估计模型的参数，确定模型的$p,q$等。
在上一篇文章《攻略推荐流程简介》中已经对$MLE$与$MAP$在应用逻辑回归来预测广告点击率中略作介绍，这里进一步对这两者进行统一。
先验分布：$p(\theta)$
似然函数:   $p(D|\theta)$
后验分布：$p(\theta|D)$
其中$\theta$为参数，$D$为数据集。他们之间的关系：
$$p(\theta|D)=\dfrac{p(\theta)\cdot p(D|\theta)}{p(D)}\tag{35}$$
#### 最大似然估计
最大后验估计就是指假设数据符合某个模型，但参数未知，然而当前$D$已经为既定事实，则希望寻求参数$\theta$使“似然”当前事实的可能性最大。即如下：
$$\theta_{MLE}=arg \underset{\theta}{max}\ p(D|\theta) \tag{36} \\
\hspace {19mm}=arg \underset{\theta}{max}\ \underset{i=1}{\Pi}p(D_i|\theta)  \\
\hspace {21mm}=arg \underset{\theta}{max}\ \sum_{i=1}ln p(D_i|\theta)  \\
$$
#### 最大后验估计
最大后验是指最大化$p(\theta|D)$, 意指在当前$D$情况下，结合参数$\theta$的先验分布和似然函数$p(D|\theta)$，估计参数$\theta$，
$$\theta_{MAP}=arg \underset{\theta}{max}\ p(\theta|D) \tag{37}  \\
\hspace {19mm}=arg \underset{\theta}{max}\ p(\theta)\cdotp(D|\theta)  \\
\hspace {30mm}=arg \underset{\theta}{max}\ \underset{i=1}{\Pi}p(D_i|\theta) \cdot p(\theta)  \\
\hspace {44mm}=arg \underset{\theta}{max}\ \sum_{i=1}lnp(D_i|\theta)+ lnp(\theta) \\
$$

#### MAP与MLE相统一
如果假设先验一无所知，$p(\theta)$为均匀分布，则$lnp(\theta)$为常数，此时二者等价。

##   线性时间序列分析
>引言：一般教材对时间序列的分析方法划分标准是：是否平稳。这里划分的标准是：是否线性。主要原因有两个：1.实际业务数据大部分是非平稳的，潜在包含增长性、周期性等；2.伴随深度学习日趋火热，RNN、LSTM等递归性甚至CNN在时间序列预估方面也有尝试。因此，我们我们将分析方法划分为线性和非线性。在前者中，我们简要介绍传统的时序分析经典方法ARIMA，在后者中，我们简要介绍LSTM的理论。

### 自回归模型AR
$AR$模型是以以前$p$期序列值$x_{t-1},x_{t-2},...,x_{t-p}$为自变量，随机变量$X_t$的取值$x_t$为因变量建立得到的线性回归模型。
$$\forall t,x_t=a_0+\sum_{i=1}^pa_ix_{t-i}+ \epsilon_t     \tag{38}$$
- 其中，$\epsilon_ts$ 是均值为0，标准差为$\sigma$的随机变量；
- $a_1,...,a_p$为自回归系数；
- $x_t$为观察的随机变量。

对满足平稳性条件的$AR(p)$模型的方程，两边取期望，得：
$$E(x_t)=E(a_0+a_1x_{t-1}+a_2x_{t-2}+...+a_px_{t-p}+ \epsilon_t )\tag{39}$$
已知$E(x_t)=\mu,E(\epsilon_t)=0$，因此：
$$u=a_0+a_1u+a_2u+...+ a_pu$$
从而得到：
$$\mu=\dfrac{a_0}{1-a_1-a_2-...-a_p}\tag{40}$$
- 均值
若$1-a_1-a_2-...-a_p \neq 0$，则$x_t$的均值存在。$x_t$的均值为0，当且仅当$a_0=0$

- 方差
平稳$AR(p)$模型的方差有界，等于常数。

- $ACF$
平稳$AR(p)$模型的自相关系数$\rho_k$呈指数速度衰减，始终有非零值，不会再$k$大于某个常数滞后就恒等于0，表明$\rho_k$具有拖尾性

- $PACF$
平稳$AR(p)$模型的偏自相关系数具有截尾性。

$AR(p)$模型的自相关系数具有拖尾性以及偏自相关系数具有截尾性是模型识别的重要依据。

### 滑动平均模型MA

$MA$模型是指随机变量$X_t$的取值$x_t$与以前各期的序列值无关，建立$x_t$与前$q$期的随机扰动$\epsilon_{t-1}, \epsilon_{t-2},...,\epsilon_{t-q}$得到的线性回归模型。
$$\forall t,x_t=\sum_{j=1}^qb_j\epsilon_{t-j}+ \epsilon_t     \tag{41}$$
- 其中，$b_1,...,b_q$为移动平滑系数。
- $E(\epsilon_t)=0;Var(\epsilon_t)=\sigma^2_\epsilon;E(\epsilon_t\epsilon_s)=0,\forall s\neq$t
由于$MA(q)$仅有白噪声过程线性组合，因此有：
$$E(x_t)=\mu$$
$$Var(x_t)=r_0=(1+\theta^2_1+\theta^2_2+...+\theta^2_q)\sigma^2_\epsilon$$
$$\begin{equation}  
\rho_l=
\left\{  
     \begin{array}{lr}  
    1,\hspace {67mm} l=0& \\
   \dfrac{\theta_l+\theta_{l+1}\theta_1+\theta_{l+2}\theta_2+...+\theta_{q}\theta_{q-l}}{(1+\theta^2_1+\theta^2_2+...+\theta^2_q)},\forall l=1,2,...q $\\
   0,\hspace {67mm} l=0 \forall l>q
    \end{array}  
\right.  
\tag{42}
\end{equation}  
$$
这表明$MA(q)$具有以下性质：
-$ACF$
自相关系数q阶截尾,即$q$阶以后的$MA(q)$模型的自相关系数马上截止，$q+1$阶就等于0。
-$PACF$
几何型或者振荡型。

### 自回归滑动平均模型ARMA

随机变量$X_t$的取值$x_t$不仅与以前$p$期的序列值有关，还与前$q$期的随机扰动有关。
$$\forall t,x_t=\sum_{i=1}^p\phi_ix_{t-i}+\sum_{j=1}^q\theta_j\epsilon_{t-j}+ \epsilon_t     \tag{43}$$
很显然，相对于$AR(p)$和$MA(q)$模型，$ARMA(p,q)$模型更具有普适性。并且$AR(p)$是$q=0$的$ARMA(p,q)$模型，$MA(q)$是$p=0$的$ARMA(p,q)$模型

在模型识别与估计中，需要决定$p$和$q$的值，选出相对最优的模型结构。通过时间序列的自相关函数$ACF$以及$PACF$大致决定。

|模型|$ACF$|$PACF$|
|:---:|:--:|:--:|
|$AR(p)$|拖尾（几何型或者振荡型）|$p$阶截尾|
|$MA(q)$|$q$阶截尾|拖尾（几何型或者振荡型）|
|$ARMA(p,q)$|拖尾（几何型或者振荡型）|拖尾（几何型或者振荡型）|

如果通过观察序列的$ACF$和$PACF$来判断$p$和$q$的值不是很明确，可以尝试建立多个模型，然后通过$AIC$或则$BIC$指标来选择。

### 单位根非平稳性与ARIMA

许多非平稳序列差分后会显示出平稳序列的性质，称这个非平稳序列为差分平稳序列。对差分平稳序列可以使用$ARIMA$模型进行拟合。
非平稳时间序列经过差分之后得到平稳的时间序列，然后通过上述的$ARMA$来建模，假设$x_t^{\prime}$是差分$d$次得到的平稳序列，其表达式如下：
$$\forall t,x_t^{\prime}=\sum_{i=1}^p\phi_ix_{t-i}^{\prime}+\sum_{j=1}^q\theta_j\epsilon_{t-j}+ \epsilon_t     \tag{44}$$
一般称该模型为$ARIMA(p,d,q)$，其中：
$p$=自回归的阶数
$d$=差分的阶数
$q$=滑动平滑的阶数
结合之前的滞后算子，则上述表达式如下：
$$(1-\phi_1L-...-\phi_pL^p)\   (1-L)^d \ x_t=\mu+(1+\theta L+...+\theta_q L^q)\epsilon_t \tag{45}$$
其中$(1-\phi_1L-...-\phi_pL^p)$表示序列的$p$阶自回归$AR(p)$特性，(1-L)^d表示序列的$d$阶差分平稳$I(d)$特性，$(1+\theta L+...+\theta_q L^q)$表示序列的$q$阶滑动平均$MA(q)$特性.

### 季节模型与SARIMA

全称为$Seasonal\  Auto \ Regressive\  Integrated\  Moving\  Average$，季节性差分自回归滑动平均模型。运用$ARMA$模型的前提条件是时间序列为零均值的平稳随机过程。对于包含趋势性或季节性的非平稳时间序列，须经过适当的逐期差分及季节差分消除趋势影响后，在对形成的新的平稳序列建立$ARMA(p,q)$模型进行分析。对于只包含趋势性的原序列，可以表示为$ARIMA(p,d,q)$模型；若原序列同时包含季节性和趋势性，则可以表示为$ARIMA(p,d,q)(P,D,Q)_s$模型，其中的$d,D$分别为逐期差分和季节差分的阶数，$p,q$分别为自回归和滑动平均的阶数，$P,Q$分别为季节自回归和季节移动平均的阶数。

## 非线性时间序列分析
>引言：近些年来，深度学习除了在文本、语音、图像上面有很大的突破。以RNN、LSTM为代表的循环神经网络被引入自然语言处理以及时间序列分析，关于LSTM及其演化可以参考[《Long short-term memory》](http://www.bioinf.jku.at/publications/older/2604.pdf)、[《A Critical Review of Recurrent Neural Networks for Sequence Learning》](http://zacklipton.com/media/papers/recurrent-network-review-lipton-2015v2.pdf)以及[《Understanding LSTM by Colah》](http://colah.github.io/posts/2015-08-Understanding-LSTMs)

### LSTM
LSTM的经典网络结构图如下：

![LSTM网络结构图](/img/lstm.jpg)

其中涉及三个门，分别如下：

forget gate:
$$f_t=\sigma(W_f\cdot[h_{t-1},x_t])+b_f \tag{46}$$
input gate:
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    i_t= \sigma(W_i\cdot[h_{t-1},x_t]+b_i) & \\
    \tilde{C_t}=tanh(W_C\cdot[h_{t-1},x_t]+b_C)
    \end{array}  
\right.  
\tag{47}
\end{equation}  
$$

output:
$$C_t=f_t*C_{t-1}+i_t*\tilde{C_t} \tag{48}$$
若令$x_c{t}=[x_t,h_{t-1}]$，则有如下前向网络：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    f_t= \sigma(W_f\cdot [x_t ,h_{t-1}]+b_o) & \\
    i_t= \sigma(W_i\cdot [x_t ,h_{t-1}]+b_o) & \\
    g_t= tanh(W_g\cdot [x_t ,h_{t-1}]+b_o) & \\
    o_t= \sigma(W_o\cdot [x_t ,h_{t-1}]+b_o) & \\
    s_t= s_{t-1} \cdot f_t+ i_t \cdot g_t & \\
    h_t=o_t*tanh(s_t)
    \end{array}  
\right.  
\tag{49}
\end{equation}  
$$
目标函数为：
$$l(t)=\dfrac{1}{2}(h_t-y_t)^2 \tag{50}$$
$$L=\sum_{t=1}^{T}l(t)=\dfrac{1}{2}\sum_{t=1}^{T}(h_t-y_t)^2 \tag{51}$$
另外激活函数
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    \sigma(x)= \dfrac{1}{1+e^{-x}} \\
    tanh(x)=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
    \end{array}  
\right.  
\tag{52}
\end{equation}  
$$
其对应的导数是：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    \sigma'(x)= x \cdot (1-x) \\
    tanh'(x)= 1-tanh^2(x)
    \end{array}  
\right.  
\tag{53}
\end{equation}  
$$
关于LSTM的文献汗牛充栋，然而各个文献的变量标记大都各有差异，在前向传播和反向梯度计算的理论以及实现细节上面都鲜有完整，为了更好理解LSTM的前向网络传播以及反向梯度计算，展开其结构，并以电路图结构展示如下：

![LSTM电路图](/img/LSTM电路图.jpg)

如上图，在前向网络传播时，输入为当前时刻的输入$x_t$以及上一个时刻的输出$h_{t-1}$构成，令：$$x_c(t)=[x_t,h_{t-1}] \tag{54}$$
其中$c$表示$concat$连接之意。
图中黄色区域中可以理解为通过1个全连接层，然后级联4个激活函数，从而得到$f_t、a_t、i_t、o_t$,表达式如下：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
    f(t)= \sigma(W_{fx}x_t +W_{fh}h_{t-1}+b_f) & \\
    a(t)= \tanh(W_{ax}x_t +W_{ah}h_{t-1}+b_a) & \\
    i(t)= \sigma(W_{ix}x_t +W_{ih}h_{t-1}+b_i) & \\
    o(t)= \sigma(W_{ox}x_t +W_{oh}h_{t-1}+b_o) & \\
    s_t= s_{t-1} \cdot f_t+ i_t \cdot a_t & \\
    h_t=o_t*tanh(s_t)
    \end{array}  
\right.  
\tag{55}
\end{equation}  
$$
可以图示理解如下：
$$
\begin{gather*}
\begin{bmatrix} x \\ h \end{bmatrix}\quad
\rightarrow
\begin{bmatrix} W_{fx} & W_{fh} \\ W_{ax} & W_{ah} \\ W_{ix} & W_{ih} \\ W_{ox} & W_{oh} \\  \end{bmatrix} \quad
\begin{bmatrix} x \\ h \end{bmatrix}\quad
+
\begin{bmatrix} b_f \\ b_a\\b_i\\b_o \end{bmatrix}\quad
\rightarrow
\begin{bmatrix} \sigma \\ \phi\\ \sigma\\\sigma \end{bmatrix}\quad
\rightarrow
\begin{bmatrix} f(t) \\ a(t)\\ i(t)\\o(t)\end{bmatrix}\quad
\end{gather*}
$$
记：
$$
\begin{gather*}
W_x=\begin{bmatrix}W_{fx} \\W_{fax}\\ W_{ix}\\W_{ox}\end{bmatrix},\quad
W_h=\begin{bmatrix}W_{fh} \\W_{fah}\\ W_{ih}\\W_{oh}\end{bmatrix},\quad
gates_t=\begin{bmatrix} f(t) \\ a(t)\\ i(t)\\o(t)\end{bmatrix}\quad
\end{gather*}
$$
### BPTT反向传播算法
根据链式求导法则，因此可以得到反向传播的梯度BPTT算法，得到针对$h_t$的导数如下：
$$ \dfrac{d_{L(t)}}{d_{h(t)}} = \dfrac{d_{l(t)}}{d_{h(t)}} +\dfrac{d_{L(t+1)}}{d_{h(t)}}\tag{56}$$
其中第一项比较简单：
$$ \dfrac{d_{l(t)}}{d_{h(t)}}=h(t)-y(t) \tag{57}$$
第二项的计算为：
$$\dfrac{d_{L(t+1)}}{d_{h(t)}}=W_h^T\delta gates_t \tag{58}$$
其中：
$$
\begin{gather*}
\delta gates_t=\begin{bmatrix} f'(t) \\ a'(t)\\ i'(t)\\o'(t)\end{bmatrix}\quad
\tag{59}
\end{gather*}
$$
另外对$s(t)$的依赖关系如下图：

![依赖关系图](/img/LSTM级联.jpg)

则其导数为：

$$
\begin{equation}  
\left .   
     \begin{array}{lr}  
     \dfrac{d_{L(t)}}{d_{s(t)}} = \dfrac{d_{l(t)}}{d_{s(t)}} +\dfrac{d_{L(t+1)}}{d_{s(t)}}&\\
                 \hspace {12mm}= \dfrac{d_{l(t)}}{d_{s(t)}} +\dfrac{d_{L(t+1)}}{d_{s(t)}}&\\
			    \hspace {12mm}= \dfrac{d_{l(t)}}{d_{h(t)}} \cdot  \dfrac{d_{h(t)}}{d_{s(t)}}+\dfrac{d_{L(t+1)}}{d_{s(t+1)}} \cdot \dfrac{d_{s(t+1)}}{d_{s(t)}}  &\\
			    \hspace {12mm}= \dfrac{d_{l(t)}}{d_{h(t)}} \cdot o_t \cdot (1-tanh^2(s_t))+\dfrac{d_{L(t+1)}}{d_{s(t+1)}} \cdot f_t
    \end{array}  
\right.  
\tag{60}
\end{equation}  
$$
上式也可以看出$\dfrac{d_{L(t)}}{d_{s(t)}}$具有递归特性。
当得到$h_t、s_t$的导数递归表达形式之后，其他的$f_t、a_t、i_t、o_t$的梯度表达式易得，整体如下：
$$
\begin{equation}  
\left\{  
     \begin{array}{lr}  
   \dfrac{d_{L(t)}}{d_{h(t)}} = \dfrac{d_{l(t)}}{d_{h(t)}} +\dfrac{d_{L(t+1)}}{d_{h(t)}} &\\
   \hspace {12mm}=\dfrac{d_{l(t)}}{d_{h(t)}} +W_h^T\delta gates_t &\\
   \dfrac{d_{L(t)}}{d_{s(t)}} = \dfrac{d_{l(t)}}{d_{s(t)}} +\dfrac{d_{L(t+1)}}{d_{s(t)}} &\\
   \hspace {12mm} = \dfrac{d_{l(t)}}{d_{h(t)}} \cdot o_t \cdot (1-tanh^2(s_t))+\dfrac{d_{L(t+1)}}{d_{s(t+1)}} \cdot f_t &\\
   \dfrac{d_{L(t)}}{d_{f(t)}}=\dfrac{d_{L(t)}}{d_{s(t)}} \cdot \dfrac{d_{s(t)}}{d_{f(t)}} &\\
     \hspace {12mm} =\dfrac{d_{L(t)}}{d_{s(t)}} \cdot s_{t-1} \cdot f_t \cdot(1-f_t) &\\
\dfrac{d_{L(t)}}{d_{a(t)}}=\dfrac{d_{L(t)}}{d_{s(t)}} \cdot \dfrac{d_{s(t)}}{d_{a(t)}} &\\
     \hspace {12mm} =\dfrac{d_{L(t)}}{d_{s(t)}} \cdot i_t \cdot(1-a^2_t) &\\
 \dfrac{d_{L(t)}}{d_{i(t)}}=\dfrac{d_{L(t)}}{d_{s(t)}} \cdot \dfrac{d_{s(t)}}{d_{i(t)}} &\\
     \hspace {12mm} =\dfrac{d_{L(t)}}{d_{s(t)}} \cdot a_t \cdot i_t \cdot(1-i_t) &\\
      \dfrac{d_{L(t)}}{d_{o(t)}}=\dfrac{d_{L(t)}}{d_{h(t)}} \cdot \dfrac{d_{h(t)}}{d_{o(t)}} &\\
     \hspace {12mm} =\dfrac{d_{L(t)}}{d_{h(t)}} \cdot tanh(s_{t}) \cdot a_t \cdot o_t \cdot(1-o_t) &\\
    \end{array}  
\right.  
\tag{61}
\end{equation}  
$$
除了在线性模型中介绍的通过分解来预测，我们也在探索应用LSTM以及CNN的方法来做时间序列分析，后续文章也将逐渐展开非线性方法的应用效果。
## 业界方案
>针对非平稳时间序列的分析，业界已经开源了不少工具，主要是R语言算法包，例如STL、Forecast、Prophet、bsts等，下面简要介绍其中Forecast和Prophet。

### Forecast
提到时间序列分析，不得不提STL和Forecast，这是在R语言中很易用的功能，为一些数学家和统计学家所钟爱。下面稍微介绍下：
- LOWESS
局部加权回归，全称为$LOcally\  wEight\ Scatterplot \ Smoothing(LOESS)$,利用低次多项式来拟合在数据集上的每个点，距离越近的点，权重就越高，相反距离越远，权重就越低。
- STL
全称为$Seasonal-Trend\ decomposition\ procedure \ based \ on \ Loess$,通过内外两层循环将时间序列分解为趋势项、季节项以及残余项。STL的流程如下：
```cpp
outer loop：
	计算robustness weight；
	inner loop：
		Step 1 :去趋势；
		Step 2: 周期子序列平滑；
		Step 3: 周期子序列的低通量滤波；
		Step 4: 去除平滑周期子序列趋势；
		Step 5: 去周期；
		Step 6: 趋势平滑；
```
为了使得算法具有足够的鲁棒性，所以设计了内循环和外循环。如果时间序列中有异常点，则余项会比较大。假设数据点$v$的余项为$R_v$,则定义：
$$h=6*median(|R_v|)  \tag{62}$$
该点的$Robustness \ Weight$为:
$$\rho_v=B(|R_v|/h) \tag{63}$$
 其中$B$函数为$bisquare$函数：
 $$
 B(u)=
 \begin{equation}  
\left\{  
     \begin{array}{lr}  
   (1-u^2)^2 \ \ for \ u\in[0,1)\\
   0 \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ for\ u\in[1,+\infty)
    \end{array}  
\right.  
\tag{64}
\end{equation}  
$$			
可以应用R语言中的STL函数方便得到时间序列的各个成分以及异常点。

- Forecast
这是一个R语言预测包，由澳大利亚的$Monash$大学的$Rob\ Hyndman$教授于2007年实现的。
这里主要举一个多周期傅里叶谐波回归的例子：
```r
msts_data<- msts(x, seasonal.periods=c(7,365)
msts_predict <- forecast(fit, xreg=fourier(msts_data, K=c(3,10), h=120))
```
其中$K$指定的为提取的傅里叶谐波级数，$h$为预测的时间跨度，具体理论可以参考[Seasonal periods](https://robjhyndman.com/hyndsight/seasonal-periods/)。
> 实际应用过程中需要注意如下几点：
 - 关于frequency的设置
 对于一个时间序列$ts$,如果没有制定freq参数，则会认为无周期项。

 - 异常点检测
 时间序列可能存在异常数据，STL分解时会除了返回各个构成项之外，同时返回各个时间点的权重，如果权重特别小，例如低于$10^{-6}$,表明该点在局部加权回归中意义不大，通常可以认为是异常点。

###  Facebook
在2017年2月份，Facebook宣布开源一款基于Python和R语言的时间序列预测工具$Prophet$,取“先知”之意。$Prophet$相比于现有预测工具更加人性化，并且同时支持Python和R语言，这一点尤其难能可贵，其项目主页[]()以及论文[]()。其中的"at scale"表明作者的初衷是想帮助大量的分析师在大量的场景和业务解决数据的预测。
$Prophet$的出发点是将时间序列分解成趋势项、周期项，同时考虑了节假日以及潜在突变量的影响。
$$y(t)=g(t)+s(t)+h(t)+\epsilon_t \tag{65}$$
针对趋势项$g(t)$，表征的是时间序列的长期趋势，$Prophet$提供了两种典型的增长函数：
- 线性：
$$g(t)=kt+b  \tag{66}$$
- 非线性：
$$g(t)=\dfrac{C}{1+exp(-k(t-b))}  \tag{67}$$
进一步考虑到业务数据可能会由于业务本身版本发布、重大事件等因素，以致于序列数据短期内发展趋势有所变化，$Prophet$为此引入了“转变点”（$change\ point$），目的是自动检测趋势的变化，表达式分别如下：
- 线性：
$$g(t)=(k+a(t)^T\delta)+(b+a(t)^T\gamma)  \tag{68}$$
- 非线性：
$$g(t)=\dfrac{C(t)}{1+exp(-(k+a(t)^T\delta)(t-(b+a(t)^T)\gamma))} \tag{69}$$
另外业务数据在每年中的各个月份、每周中的各天、全年的各个节假日都会有不同的表现，$Prophet$通过采用傅里叶谐波级数来表达每年各个月波动以及每周中的各天波动，通过虚拟变量(dummy variables)来处理用户自己设置的重要节日，表达式分别如下：
- 线性：
$$y(t)=(k+a(t)^T\delta)+(b+a(t)^T\gamma)+X\beta+\epsilon_t \tag{70}$$
- 非线性：
$$y(t)=\dfrac{C(t)}{1+exp(-(k+a(t)^T\delta)(t-(b+a(t)^T)\gamma))}+X\beta+\epsilon_t \tag{71}$$
其中各个变量分别表示：

| 变量      |    维度大小 | 含义  |
| :-------- | --------:| :--: |
| $t$  | 1*T |  代表时间序列数据，总长度为T  |
| $k$     |   1*1 |  实数，表示增长率  |
| $b$      |    1*1 | 实数，表示截距  |
| $\delta$      |    S*1 | 向量，表示增长率的变化  |
| $\gamma$      |    S*1 | 向量，表示截距的变化  |
| $a$      |    T*S | 矩阵，表示序列趋势的回归系数 |
| $\beta$      |    K*1 | 向量，表示周期项以及节假日项回归系数  |
| $X$      |    T*K | 矩阵，表示周期项以及节假日项目的矩阵  |
| $\epsilon$      |    T*1 | 向量，表示模型拟合残差  |

$Prophet$结合先验知识，假设数据服从线性或非线性两种似然函数（likelihood function），另外对参数$k,b,\delta,\beta,\epsilon$也给以先验分布假设，通过最大后验证估计来得到满足当前假设的最有参数，从而进行预估。

> 实际应用过程中需要注意如下几点：
	- Capacity的选择
	需要结合实际业务数据设置，如果设置过大，预估数据会偏高。
	- 节假日的设置
	需要分析节假日属性，一般相同属性的节日归为一类，例如在视频播放数据中，除夕播放量低，暑期开学低，可以归为一类；国庆节播放量大，可以设为另一类。还要注意节假日的持续时间长度设置。
	- 长期误差与短期误差
	如果考虑了长周期，则长期预估误差通常低于其他算法，然而在短期预估方面，效果一般。这个主要是由于该算法在考虑了week周期来刻画每周的变化规律，但是没有考虑自回归的影响。

总之，$Prophet$一方面简单易用，一方面由于考虑了很多先验假设，因此需要先分析实际业务数据的一些特性，例如趋势、周期特性等，再应用其来预估，往往会达到更好的预估效果。

### Ensemble
针对同一个时间序列，从信息论的角度看，好的模型应该是能够完全提取序列的潜在模式，如内在的趋势、周期变化规律以及节假日的影响等，分解之后的残差服从高斯分布。同时，这个模型应该没有因为模式和参数的先验假设或者优化求解过程而产生过拟合。那么，该模型一般可以比较好的预测未来。然而，具体业务时间数据可能存在某种波动，单一模型不能有效把握，则需要同时考虑多个模型集成，往往可以获得更加鲁棒的预估性能。


## 下一步工作
本文所述时间序列分析理论和方法依然更多关注序列本身，关注如何充分挖掘当前序列的信息，然而实际时间序列本身是一个不断演进的过程，就模型而言，还可以应用条件异方差自回归模型$GARCH$来刻画模型，或者考虑应用小波变换来提取时间序列的波动性。更重要的是，时间序列的演进还有很多外在因素，如何引入外在因素，如舆情指数、多个时间变量相关性等进一步提升预估准确率，提前发现并预警预估误差波动过大的缘由，依然需要更进一步深入分析。

## 总结
林尽水源，便得一山，山有小口，仿佛若有光。

## 致谢
感谢各位同事参与讨论指导。

## 参考文献
- [$Prophet$](https://facebook.github.io/prophet/)
- [$bsts(Bayesian\ Structual\ Time\ Series)$](https://www.rdocumentation.org/packages/bsts/versions/0.7.1/topics/bsts)
- [$Anomaly\ Detection$](https://github.com/twitter/AnomalyDetection)
- [$Robust\ Anomaly\ Detection$](https://github.com/Netflix/Surus)
- [$DBP(Discrepancy\ based\ Forecast)$](https://cs.nyu.edu/~mohri/talks/NIPSTutorial2016.pdf)
-  [$《Forecasting: principles and practice - Rob \ J.\  Hyndman$》](https://www.otexts.org/fpp/6/1)
- 《金融时间序列分析-蔡瑞胸》
- 《量化投资-以Python为工具-蔡立耑》
- 《时间序列分析-汉密尔顿》
- 《数学物理方法-胡学刚》
- 《数学分析-陈传璋》
- 《信号与系统-郑君里》
- 《数字信号处理-程培青》
- 《通信原理-周炯槃》
- 《Pattern Recognition and Machine Learning- Christopher Bishop 》
- 《The Elements of Statistical Learning-Trevor Hastie》
-  [《Long short-term memory》](http://www.bioinf.jku.at/publications/older/2604.pdf)
-  [《A Critical Review of Recurrent Neural Networks for Sequence Learning》](http://zacklipton.com/media/papers/recurrent-network-review-lipton-2015v2.pdf)
- [Understanding LSTM - Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs)
- [Web Traffic Time Series Forecasting-Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
