Contents lists available at ScienceDirect

Infrared Physics and Technology

journal homepage: www.elsevier.com/locate/infrared

Research Paper

Deep soft threshold feature separation network for infrared handprint
identity recognition and time estimation

Xiao Yu *, Xiaojie Liang , Zijie Zhou , Baofeng Zhang , Hao Xue *

Tianjin University of Technology, Tianjin, China

A R T I C L E  I N F O

A B S T R A C T

Keywords:
Infrared image recognition
Feature separation
Soft thresholding
Time estimation

With the development of hardware devices, infrared technology has become an important detection method in
criminal  investigation,  military  and  other  fields.  Infrared  technology  can  capture  the  thermal  infrared  infor-
mation of targets, achieve efficient and rapid individual identification, provide important technological support
for crime investigation, improve case solving rate and public safety level. However, infrared technology still faces
some  difficulties.  Due  to  factors  such  as  environmental  temperature  changes  and  heat  transfer,  the  infrared
images captured over time gradually blur, resulting in highly aliased image features and difficulty in extracting
effective  information.  This  article  addresses  this  issue  by  combining  the  specific  tasks  of  infrared  fingerprint
identity recognition and time estimation. By analyzing the captured thermal traces to predict the target identity
and departure time, a deep soft threshold feature separation (DSTFS) network is proposed. This network effec-
tively  improves  the  accuracy  of  identity  recognition  and  time  estimation  by  separating  the  characteristics  of
identity information and time information. Specifically, this article is based on the ResNet backbone network,
extracting common features through shallow residual convolution blocks, extracting corresponding features for
different tasks using branch residual convolution blocks, and introducing spatial and channel attention mech-
anisms to solve the problems of deep blur and feature separation. In order to enhance the model’s expressive
power, the soft threshold activation function SPRelu was adopted. By dynamically balancing the separation of
features to learn weights, the weight balancing problem is solved, and periodic validation is used to evaluate task
performance. According to the task requirements, we collected infrared fingerprint images of multiple testers and
constructed an infrared hand heat trace dataset with labels such as departure time, gender, hand posture, and
identity category as experimental data. Through the analysis of experimental results, our proposed deep feature
separation network  has better  identity prediction  results compared to other  deep learning  models in infrared
fingerprint identity recognition and time estimation tasks. At the same time, it performs well in time estimation
tasks, with lower error rates and higher stability.

1. Introduction

1.1. Background

In criminal investigation, trace testing is a widely used and crucial
key means, widely adopted by criminal police. Through in-depth trace
analysis of crime scenes and related physical evidence, criminal police
can obtain rich information and provide strong support for successfully
solving cases. This efficient and sophisticated scientific and technolog-
ical means enables the criminal police to reveal the key details of the
crime, so as to more accurately restore the process of the crime, track the

suspect, and ultimately effectively promote the successful detection of
the  case.  The  application  scope  of  trace  inspection  is  extensive  [1],
including the analysis of fingerprints, footprints, fingerprints, fibers, and
other aspects, providing comprehensive and reliable evidence basis for
criminal police and playing an indispensable role in judicial fairness.

In the field of trace inspection, extracting and analyzing fingerprint
information  from  crime  scenes  has  always  been  a  routine  criminal
investigation technique. With the continuous progress of modern crim-
inal investigation, the introduction of advanced technological means has
become  a  trend,  among  which  the  application  of  infrared  technology
[2,3] has become an efficient means to quickly capture and track heat

* Corresponding authors.

E-mail addresses: yx_tjut@163.com (X. Yu), x17538321751@163.com (H. Xue).

1  Xiao Yu is a professor at Tianjin University of Technology, with research interests in artificial intelligence, machine learning, image processing, and data mining.

https://doi.org/10.1016/j.infrared.2024.105223
Received 10 November 2023; Received in revised form 18 January 2024; Accepted 7 February 2024

InfraredPhysics&Technology138(2024)105223Availableonline12February20241350-4495/©2024ElsevierB.V.Allrightsreserved.X. Yu et al.

trace information, providing strong support for case investigation. In the
context  of  using  infrared  technology,  quickly  capturing  and  tracking
thermal trace information can provide important clues in case investi-
gation.  Whether  in  daily  touch  events  or  complex  criminal  in-
vestigations,  infrared  images  can  capture  the  thermal  distribution  of
objects in a highly sensitive manner [4]. Using infrared technology to
capture hand thermal traces and obtain biometric information related to
individual  identity,  such  as  the  temperature  distribution  of  the  palm.
These features can be used for identity recognition to build a reliable
identity  verification  system.  This  technology  provides  criminal  police
with  a  more  comprehensive  and  in-depth  perspective  on  cases.  By
combining the extraction and analysis of traditional fingerprint infor-
mation,  infrared  technology  provides  investigators  with  a  more  com-
plete  and  accurate  tool  for  case  reconstruction,  which  is  expected  to
accelerate the process of solving cases and become an efficient means of
case investigation.

By identifying heat marks on the hands, it can not only be used for
identity recognition, but also assist in estimating the time of the marks.
Infrared fingerprint images contain time information. By analyzing the
time  characteristics  of  hand  heat  traces,  the  departure  time  after  the
traces  are  generated  can  be  estimated.  By  establishing  a  time  related
heat  trace  model,  the  approximate  range  of  departure  time  can  be
inferred,  improving  the  understanding  of  the  timeline  of  the  crime
scene.  By  comprehensively  identifying  identity  and  estimating  trace
time,  criminal  investigators  can  have  a  more  comprehensive  under-
standing of the situation at the crime scene. This comprehensive analysis
helps to build a more accurate and complete case reconstruction, and is
of great significance in narrowing the search scope of suspect.

This article proposes a deep feature separation infrared fingerprint
identity recognition and trace time estimation method, which simulta-
neously  extracts  fingerprint  information  and  time  information  from
infrared  images.  This  method  not  only  improves  identity  recognition
performance, but also has more reliable individual identity confirma-
tion. At the same time, the ability to estimate trace departure time also
helps to more accurately restore the timeline of the crime process. These
two  aspects  of  information  work  together  to  provide  more  powerful
evidence for criminal investigation, help solve cases and track suspect.

1.2. Research objective

The aim of this study is to achieve precise recognition and trace time
estimation of infrared fingerprint images, providing a more efficient and
reliable means for criminal investigators to cope with complex and ever-
changing crime scenes, providing strong support for the resolution of
criminal cases, and providing strong technical support for the judicial
system, thus making positive contributions to social security and public
security maintenance.

The study aims to construct a deep learning model that can deeply
separate features to achieve dual feature extraction of identity and time
for infrared fingerprints, while completing identity recognition and time
estimation  tasks.  By  establishing  a  hand  heat  trace  database  and
recording the composition and characteristics of relevant data in detail,
a sufficient experimental basis is provided for model training. Design a
deep learning model to effectively extract identity and temporal infor-
mation from hand thermal traces, and achieve collaborative optimiza-
tion between the two through a deep feature separation framework.

By comparing the performance of classic deep learning models, it can
be concluded that there are certain limitations in conducting infrared
fingerprint identity recognition tasks alone. By using feature separation
methods to simultaneously extract identity and time features, and syn-
chronously training the two tasks, the performance of the model can be
improved, making it more outstanding in complex criminal investiga-
tion  scene  traces.  In  addition,  considering  the  noise  interference  of
thermal  traces,  a  soft  threshold  activation  function  is  introduced  to
optimize  the  basic  model  and  improve  the  overall  robustness  of  the
model.

By comparing and conducting ablation experiments, we studied the
performance of our model in infrared fingerprint identity recognition. At
the same time, we conducted residual analysis on the results of depar-
ture time estimation to gain a deeper understanding of the limitations of
the model in handling trace targets with different departure times, and
to  explore  the  characteristics  of  the
learned  time  estimation
performance.

1.3. Research status

Research  on  infrared  thermal  traces  has  made  certain  progress,
especially in the fields of military, criminal investigation, medicine, and
security. Researchers have successfully captured and analyzed thermal
traces  generated  by  object  thermal  radiation  or  contact  with  objects
using tools  such as  thermal imagers,  thus  achieving multiple applica-
tions. In terms of military target detection [5], researchers have intro-
duced  a  fourth-order  feature  extraction  layer  to  inspect  the  detected
infrared images, effectively integrating infrared information of different
scales, and improving the accuracy and robustness of target detection. In
the  field  of  criminal  investigation,  infrared  thermal  trace  technology
reveals hidden clues at the crime scene by detecting small temperature
differences, helping criminal investigators track suspect. In the medical
field,  infrared  technology  can  automatically  analyze  the  temperature
distribution of areas of interest and perform statistical analysis to detect
anomalies  [6].  In  tumor  diagnosis  and  treatment  monitoring,  by
measuring the infrared radiation of human tissues, scientists can identify
abnormal heat traces and improve the accuracy of early disease diag-
nosis.  In  the  field  of  security,  researchers  have  proposed  a  password
cracking  technique  based  on  thermal  imaging,  which  uses  physical
models and inversion algorithms to crack passwords [7]. Research has
shown that the accuracy of using thermal imaging technology to crack
passwords is relatively high, posing new challenges to password secu-
rity. Some researchers have also explored the feasibility of using heat
tracing technology to unlock passwords and unlock phone modes [8].
The  research  results  indicate  that  even  if  the  password  is  repeatedly
unlocked, the accuracy of cracking is higher than 72 %. These studies,
similar to other related fields, focus on estimating the time of thermal
traces or determining the order of trace generation.

In addition to the research on infrared technology mentioned above,
some  scholars  have  also  paid  attention  to  the  application  of  thermal
trace mining for detailed information. They used thermal imaging and
hidden information testing to detect the deceptive behavior of simulated
criminal participants [9], further expanding the application of thermal
tracing technology in criminal investigation scenes. The development of
simulation models enables effective simulation of heat dissipation pro-
cesses  in  real  scenarios  with  different  contact  materials  and  contact
times [10], providing powerful tools for the study of practical cases.

Unlike  previous  studies,  the  research  mentioned  in  this  article  fo-
cuses on predicting hand identity and estimating departure time over
long time. By using deep learning methods, hand trace features can be
extracted  from  heat  trace  images  taken  at  the  crime  scene,  enabling
identity prediction and estimation of departure time.

2. Infrared fingerprint identification and trace time estimation

2.1. The problem of identity recognition task under mixed features

The infrared fingerprint identity recognition task, as a classic image
classification problem, faces a series of challenges in the application of
traditional  image  classification  models  due  to  its  unique  properties.
Although  the  development  of  image  classification  models  has  a  long
history,  in-depth  research  and  exploration  are  still  needed  for  the
recognition  of  infrared  fingerprints.  Traditional  image  classification
networks  have  not  shown  satisfactory  detection  results  in  infrared
fingerprint  identity  recognition,  which  not  only  tests  the  model’s
generalization ability, but also involves the extraction and modeling of

InfraredPhysicsandTechnology138(2024)1052232X. Yu et al.

unique information from infrared fingerprints.

To study the performance of classic image classification models in
infrared fingerprint identity recognition tasks, this paper first constructs
a  dataset  specifically  for  infrared  fingerprints  and  selects  a  series  of
classic  image  classification  models,  including  Alexnet  network  [11],
deep  convolutional  network  (VGG)  [12],  deep  residual  network
(ResNet) [13], as well as Efficientnet [14], Regnet network [15], Goo-
gLeNet network [16], and VisionTransformer model [17]. These models
perform well in traditional visible light image classification tasks, but
their application in the field of infrared fingerprint has not yet reached
the ideal level.

For the latest infrared images with clear fingerprints, these classic
image classification networks can achieve excellent identity recognition
results. However, with the addition of infrared fingerprint images from
multiple time periods in the dataset, these images gradually blur and
fade over time, and the recognition performance of classic image clas-
sification models becomes worse and worse.

The infrared fingerprint image dataset from all time periods was used

for model training, and the final test results are shown in Table 1.

Through experimental results, it can be found that even though the
network model has converged during training and the loss function is
difficult to further decrease, the overall recognition performance of the
model is still poor, making it difficult to achieve the ideal accuracy of
fingerprint identity recognition. It can be seen that in infrared finger-
print identity recognition, there must be some interference factor, which
makes  the  features  between  identity  categories  highly  aliased  and
difficult to effectively extract. During the experiment, the recognition
accuracy of the model ultimately decreased with the increase of infrared
fingerprint shooting time. Therefore, it is speculated that under the in-
fluence  of  thermal  diffusion,  the  trace  information  of  infrared  finger-
prints  gradually  became  blurred,  ultimately  leading  to  difficulty  in
extracting identity features.

2.2. Current problems faced

The main problem faced in infrared fingerprint identity recognition

tasks is the depth blur and time invariance of thermal traces [18].

The problem of blurred thermal traces is that with the increase of
infrared fingerprint shooting time, due to the reflection and scattering of
infrared light on the surface of hot objects, and the temperature change
caused by thermal conduction, the trace information may be gradually
blurred due to the influence of thermal diffusion. Due to the blurring of
infrared  fingerprint  trace  information,  key  features  in  the  image  may
become difficult to extract. Image classification models typically rely on
local and global features in the image to identify objects or identities. If
the trace information is unclear, the model may not be able to accurately
capture identity features, thereby affecting classification performance.
In  addition,  thermal  diffusion  may  cause  changes  in  image  quality,
thereby affecting the performance of the model. The decrease in image
quality may include factors such as reduced contrast and loss of details,
all of which can have a negative impact on model training and inference.
The problem of heat trace blurring is related to the final recognition
performance of image classification models. However, there is relatively
little research on this problem in infrared fingerprint identity recogni-
tion  tasks.  Fortunately,  we  can  find  a  similar  problem  in  facial

Table 1
The Performance of Classic Models in Identity Recognition.

Model

Params(M)

Accuracy (%)

Alexnet
VGG16
Resnet18
Efficientnet-b7
Regnet_x_400mf
GoogLeNet
Vision Transformer

57.01
134.28
11.17
63.80
5.10
6.25
9.56

58.04
64.73
74.06
66.35
68.69
64.59
69.52

recognition tasks, namely the time invariance problem [19–22]. In the
field of facial recognition, time invariance refers to the ability of algo-
rithms  to  accurately  recognize  the  same  facial  image  collected  at
different time points, without being affected by time. Time invariance in
facial  recognition  mainly  includes  light  invariance,  facial  expression
invariance, posture invariance, and age invariance. From the similarity
of actual problems, it can be seen that the age invariance problem in
facial recognition is also based on the problem of image feature changes
caused by the passage of time.

In cross age facial recognition tasks, numerous studies have found
that features such as wrinkles appear during facial aging, leading to an
increase in intra class spacing and significant intra class differences. This
situation is called an age invariant face recognition problem. Analogous
to the infrared fingerprint identity recognition task in this article, during
the thermal diffusion process of infrared fingerprints, features such as
fuzzy diffusion appear, which also lead to significant intra class spacing
in infrared images of unified identity fingerprints at different time pe-
riods. When the intra class distance is too high compared to the inter
class  distance  of  fingerprints,  the  network  finds  it  difficult  to  learn
effective  features  and  thus  cannot  achieve  ideal  identity  recognition
accuracy.

As  shown  in  Fig. 1,  clear  infrared  fingerprint  images have  a  large
inter  class  distance,  making  them  easily  distinguishable  by  trained
classifiers. As time goes on, the original image features gradually shift
under the influence of thermal diffusion. A2 in the figure represents the
features of A1 after a period of offset, while A3 represents the features of
A2 after a period of offset. As the time shift brings about feature changes,
it ultimately leads to the phenomenon of gradually expanding intra class
spacing in sample A. When the intra class spacing increases by more than
a certain degree, that is, the class exceeds the inter class spacing, it may
cross the classifier boundary, resulting in a decrease in recognition ac-
curacy. And the traces obtained by the same identity at different time
points will undergo greater changes due to thermal diffusion, which are
more abstract and unobservable compared to faces with age differences.
It can be inferred that the depth blur formed in all infrared finger-
prints  contains  similar  time  related  information  shared.  Therefore,  in
this  article,  we  refer  to  this  phenomenon  as  time  invariant  infrared
fingerprint recognition. Decomposing identity related information and
time related information in infrared fingerprints has become an urgent
research focus.

3. Deep soft threshold feature separation network

In infrared fingerprint images, identity related features refer to in-
dividual  identity  related  features  in  hand  marks.  In  identity  related
features,  the  model  attempts  to  capture  unique  patterns,  biological
features, and other information related to individual identity, including
hand  shape,  heat  distribution,  and  other  features  closely  related  to

Fig. 1. Intra class and inter class spacing caused by time passage.

InfraredPhysicsandTechnology138(2024)1052233X. Yu et al.

individual identity. The temporal correlation feature involves the vari-
ation  of  hand  marks  over  time.  Due  to  the  depth  blurring  caused  by
thermal diffusion, hand marks may exhibit some blurry characteristics at
different time points. The temporal correlation features aim to capture
these  changes  in  order  to  better  understand  the  evolution  process  of
hand  marks.  Based  on  the  previous  analysis,  the  performance  of  the
classic  single  task  network  model  in  infrared  fingerprint  identity
recognition  tasks  is  not  satisfactory,  mainly  due  to  the  deep  overlap
between  identity feature information  and time feature information in
infrared  fingerprint  images,  which  brings  great  difficulties  to  the
recognition task. In this case, the primary challenge our model faces is
how to effectively separate these overlapping features.

To address this issue, inspired by multi task learning networks [23],
this paper proposes a deep soft threshold feature separation method. In
multitasking  learning, the  backbone of  the network  is responsible  for
extracting  shared  features  between  different  tasks,  while  the  branch
networks dedicated to each task focus on extracting specific  features.
This feature decomposition method provides us with an effective way to
solve the problem of aliasing features.

Identity association  features and time association  features provide
different aspects of information. The relationship between these two lies
in the fact that hand traces contain both identity and temporal infor-
mation, and by simultaneously learning these two tasks, the model can
deeply explore the inherent characteristics of hand traces from multiple
aspects,  achieving  complementary  and  synergistic  effects  of  informa-
tion. Based on the complementary and collaborative features of the two,
multi task learning methods can not only improve the comprehensive
modeling  ability  of  the  model  for  hand  marks,  but  also  enhance  the
robustness of the model and better adapt to practical needs.

The key issue in feature separation networks is how to process and
separate features to ensure that the model can focus on relevant features
when  performing  different  tasks.  A  common  approach  is  to  design  a
shared  feature  extraction  layer  that  learns  common  features  that  are
useful for multiple tasks. In this way, the model can learn specific task
features required to perform each task on the basis of shared features.
This  approach  helps  to  avoid  conflicting  feature  representations  be-
tween different tasks.

When there is strong correlation between tasks, feature separation
networks are often able to better share and separate features. On the
contrary, when there are significant differences between tasks, it may be
necessary to design the shared feature extraction layer more carefully to
ensure that the model can maintain effective feature separation while
executing different tasks.  Therefore, the effectiveness  of feature sepa-
ration  networks  depends  on  the  rationality  of  task  selection,  feature
design, and model structure. The infrared fingerprint identity recogni-
tion  task  and  time estimation  task  have  a  strong  correlation,  and  the
underlying features often overlap highly. The use of separated features
can effectively improve the performance of the model.

Therefore, the deep soft threshold feature separation method treats
infrared fingerprint identity recognition and time estimation tasks as a
multi  task  learning  problem  with  separable  features.  Compared  with
single unseparated feature learning, this method shares knowledge and
features,  enabling  the  model  to  learn  and  generalize  more  effectively
between identity recognition and time estimation tasks. Ultimately, the
model’s  generalization  performance  is  improved  through  the  interde-
pendence between tasks, Enable it to better adapt to tasks.

3.1. Deep feature separation network based on ResNet

The  Deep  Feature  Separation  Network  (DSTFS)  we  propose  is  a
branch architecture that focuses on identity recognition and time esti-
mation tasks, with the core of improving recognition accuracy through
effective  feature  decomposition.  When  processing  the  overall  aliasing
features of an image, the key lies in extracting identity related features,
which lays the foundation for more accurate identity recognition. Due to
the  relative  difficulty  in  extracting  identity  feature  information

separately, the design of the network takes into account the aliasing of
fingerprint features, which mainly include identity and time features.
Due to the crucial role of time related information in the feature changes
of fingerprint images, the deep soft threshold feature separation network
adopts an innovative approach: first, extract time features, and then peel
these time features off the aliasing features to obtain identity informa-
tion features.

The key to feature separation lies in whether the model has strong
feature extraction capabilities. To achieve this goal, we choose to use the
ResNet  network  as  the  backbone  of  the  model.  The  Resnet  network
makes it easier for the network to learn complex feature representations
by introducing residual blocks and cross layer residual connections. This
design decision enables deep feature separation networks to better un-
derstand information in images and demonstrate excellent performance
in extracting temporal and identity features. The overall structure of the
deep feature separation network is shown in Fig. 2.

The backbone of the network consists of a complete ResNet network,
including  the  first  convolutional  layer  and  the  residual  convolutional
blocks  of  the  first  three  layers,  which  constitute  the  shared  feature
extraction part of the network. Subsequently, the network enters branch
structures for different tasks. When processing temporal features, depth
features are further extracted from the temporal feature map through
multiple attention modules using a set residual convolution block. These
deep  features  are  directly  output  through  fully  connected  layers  to
complete  the  task  of  time  estimation.  For  the  extraction  of  identity
features,  the  backbone  ResNet  also  extracts  deep  features  through
multiple  attention  modules.  In  order  to  effectively  remove  temporal
features,  identity  features  are  directly  subtracted  after  deep  temporal
features. This process aims to train more independent and specific deep
identity features, enabling the network to capture identity information
more accurately.

This architecture design fully utilizes the advantages of ResNet and
achieves effective extraction and separation of time and identity features
through the combination of residual convolutional blocks and multiple
attention  modules.  Overall,  this  network  structure  can  better  balance
and optimize the learning and expression of features when dealing with
different tasks.

Specifically,  the  model  takes  112  × 112  sized  infrared  palmprint
images as input. After extraction through the first convolutional layer
and three stages of residual convolutional blocks, a feature vector of size
14 × 14 × 256 is obtained. Following a branching structure, the features
are further extracted through an additional layer of residual convolu-
tional blocks, resulting in a feature map of size 7 × 7 × 512. For the time
estimation task, the features undergo an attention module and are then
regressed through a fully connected layer for palmprint time estimation.
In  the  identity  recognition  task,  the  features  extracted  from  the  time
estimation task are combined and processed through a fully connected
layer for identity recognition.

The deep feature separation network ultimately achieves the sepa-
ration of identity features and time features, linearly decomposing the
mixed features contained in the image into two unrelated parts, such as:

X = X1 + X2

X represents  the overall mixed features of infrared fingerprint im-
ages, while X1 and X2 represent identity related and time related feature
components, respectively. The decomposed features are further extrac-
ted  through  residual  convolutional  blocks,  and  attention  mechanisms

Fig. 2. The overall structure of deep feature separation network.

InfraredPhysicsandTechnology138(2024)1052234X. Yu et al.

are  used  to  supervise  the  learning  of  multiple  components,  thereby
achieving feature decomposition in deeper semantic spaces.

3.2. Multiple attention modules

The design of the multiple attention module aims to further extract
the deep information of features and pay more attention to the repre-
sentation  part  of  features.  This  module  mainly  consists  of  two  key
components,  namely  spatial  attention  and  channel  attention,  which,
through their combined effect, can more comprehensively capture the
correlation between features.

Spatial  attention  allows  the  model  to  focus  on  information  from
different positions in the input feature map. By using multiple attention
heads, the model can learn relationships at different positions in paral-
lel, thereby better capturing complex spatial structures. This enables the
multi attention module to effectively process local and global informa-
tion in the image, improving the accuracy of feature extraction. Channel
attention allows the model to focus on the correlation between different
channels  in  the  input  feature  map.  By  introducing  channel  attention
mechanism, the model can learn the contribution of each channel to the
final feature representation and dynamically adjust the weights of each
channel. This helps to enhance the sensitivity of the model to specific
features, thereby better adapting to the needs of different tasks.

In this article, a parallel combination of channel attention [24] and
spatial  attention  [25,26]  mechanisms  is  used  to  further  deepen  the
feature components at the spatial and channel levels. It can be expressed
as:

Xtime = X2*(fCA(X2) + fSA(X2) )
Xid = X1*(fCA(X1) + fSA(X1) ) (cid:0) Xtime

Among them, * represents element multiplication, fCA(⋅) and fSA(⋅)
represent  channel  attention  module  and  spatial  attention  module,
respectively. By using the attention module of time estimation task su-
pervision, time related information in the feature map can be separated,
and the remaining part can be used as identity related information. This
identity  related  information  can  be  supervised  through  hand  trace
identity recognition tasks.

The structure of the multiple attention module is shown in Fig. 3.
Firstly, by using spatial attention, weight allocation is performed on the
input feature map to capture information from different positions; Next,
weight allocation is performed on channels through channel attention to
capture the correlations between different channels. Finally, the features
adjusted by spatial and channel attention are added and merged to form
the final multi attention representation.

This  parallel structure enables multiple attention modules to have
strong expressive power and can flexibly adapt to complex feature re-
lationships. In the task of infrared fingerprint identity recognition and

time  estimation,  the  multiple  attention  module  improves  the  model’s
ability  to  abstract  and  understand  input  features  by  comprehensively
considering spatial and channel relationships, thereby extracting more
critical features to improve the model’s feature decomposition ability.

3.3. Residual convolutional structure and adaptive soft threshold module

The  residual  convolutional  blocks  in  deep  soft  threshold  feature
separation  networks  can  be  adjusted  by  setting  different  repetition
structures  and  output  channels.  The  specific  structure  of  the  residual
convolution block in the network is shown in Fig. 4.

On the left side of the figure is a residual structure with a step size of
2, which is responsible for reducing the spatial size of the feature map
and increasing the receptive field of the network. By reducing the spatial
size,  the  network  can capture  features  in  images  over  a  larger range,
thereby  processing  information  of  different  scales  and  levels  in  input
data,  better  understanding  the  spatial  structure  of  input  data,  and
improving  the  network’s  ability  to  extract  complex  patterns.  On  the
right side of the figure is a residual structure with a step size equal to 1,
which allows information to be more fully transmitted in skip connec-
tions, thereby alleviating the gradient vanishing problem in deep neural
networks. At the same time, residual blocks with a step size of 1 retain
more spatial information, making it easier for the network to learn fine-
grained  features,  which  helps  maintain  more  spatial  resolution  and
better  capture  local  structures  and  details  in  the  image.  The  specific
network  implementation  structure  of  the  residual  block  is  shown  in
Fig. 5. (a) In the figure represents the residual part with a step size of 2,
and (b) in the figure represents the residual structure with a step size of
1.

In the feature extraction process of deep convolutional neural net-
works, as the number of convolutional layers gradually increases, the
network will face a problem that the extracted features may contain a
large  amount  of  redundant  information.  This  is  because  high-level
convolutional  layers  learn  both  the  details  and  global  information  of
the input image, resulting in redundant representations in the feature
map.  This  redundancy  may  affect  the  generalization  ability  and
computational  efficiency  of  the  network.  Meanwhile,  noise  features
from  the  front  layer  network  may  be  transmitted  to  the  subsequent
layers, thereby being amplified in the stack of multi-layer convolutions.
In deep neural networks, this phenomenon may lead to the model being
overly sensitive to unnecessary details in the input data, affecting overall
performance.

To address one of the issues, a common approach is to introduce a
soft threshold region into the activation function [27,28]. The core idea
of  this  technology  is  to  filter  out  noise  by  setting  a  threshold  for  the
activation function and setting the activation values below the threshold
to zero. This operation helps to reduce redundant information in feature
mapping  and  effectively  mitigate  the  impact  of  potential  noise  from

Fig. 3. Structure of Multiple Attention Mechanisms.

Fig. 4. Residual Convolutional Block Structure.

InfraredPhysicsandTechnology138(2024)1052235X. Yu et al.

Fig.  5. Specific  network  implementation  structure of  residual  blocks (a)  rep-
resents residual blocks with a step size of 2 (b) represents residual blocks with a
step size of 1.

previous levels.

This article proposes a soft threshold activation function SPRelu to
address the issue of noise, and adds it to the residual block. Considering
that the residual part with a step size of 1 is more prone to redundant
noise after multiple stacking operations, only the residual part with a
step size of 1 is added with a soft threshold.

Soft thresholding achieves feature denoising and compressed repre-
sentation  by  setting  signal  components  smaller  than  the  threshold  to
zero. By introducing a soft threshold nonlinear function, the weights of
the model can be pushed towards sparsity, which means that weights
smaller than the set threshold are set to zero, promoting the model to
learn more compact and generalized representations. The soft threshold
activation function SPRelu image is shown in Fig. 6. On the basis of the
original  activation  function  Prelu,  a  soft  threshold  region  has  been
added.Fig. 7

The soft threshold activation function SPRelu can be expressed as:
⎧
⎨

y =

⎩

t

x (cid:0)
0
αx + t

(x > t)
( (cid:0)
(x < (cid:0) t)

t⩽x⩽t)

Where  x  is  the  input  feature,  y  is  the  output  feature,  and  t  is  the
threshold. Set negative features to slope with PReLU activation function
α The straight line is different, and the soft threshold is set to 0 based on
the threshold t for features close to zero. The specific implementation
process  of  the  soft  threshold  activation  function  is  shown  in  Fig.  3b.
Firstly, the extracted feature map is compressed using absolute opera-
tion and global average pooling operation. Then, the features are com-
pressed  into  one-dimensional  vectors  through  two  fully  connected
layers.  Finally,  the  length  of  the  feature  vector  output  by  the  fully

Fig. 6. (a) Shows the original Prelu activation function, and (b) shows the soft
threshold activation function SPRelu.

Fig. 7. Smaller numerical errors are greater under the square root function.

connected layer is the number of channels for the input feature mapping.
At the same time, the output needs to be limited to the range of 0 to 1, so
the sigmoid function was used to map the features. Subsequently, we
will  multiply  the  obtained  results  element  by  element  with  the  one-
dimensional  vector  before  inputting  the  fully  connected  layer  to
obtain  the  threshold  t.  The  obtained  threshold  t  will  be  used  as  a
parameter of SPRelu to implement the activation function.

The algorithm process of the soft threshold activation function is as

follows:

Algorithm: Soft thresholding activation function

Input: Input feature x
Step 1:

Perform absolute value and global pooling operations on input features.

Step 2:

Redundant information discrimination, using fully connected layers to determine
redundant information, and setting threshold coefficients using sigmoid function.

Step 3:

Multiplying the threshold coefficient with the features before full connection to
obtain the threshold.

Step 4:

Set activation function based on threshold.

Output: output feature y

3.4. Loss function design

How to train ideal feature components in a targeted manner is a key
issue that must be addressed in feature decomposition. In multitasking
infrared  fingerprint  identity  recognition  and  time  estimation  tasks,
training  corresponding  to  identity  related  features  and  time  related
features. In order to extract feature components with high stability and
quality,  targeted  supervision  methods  need  to  be  set  up  for  different
tasks.  In  this  article,  supervisory  functions  are  designed  for  identity
recognition and time estimation tasks, respectively. The time estimation
task  performs  trace  features  that  change  over  time,  while  identity
related components encode identity related information. For two tasks,
this article designs two classifiers, classifier 1 corresponds to the time
estimation task, and classifier 2 corresponds to the identity recognition
task.

Classifier 1 flattens the time-related features output by the residual
convolution block, further extracts features through two linearly fully
connected layers, and outputs them as estimates of time. For the loss
function of time, the initial design used a simple mean square error to
represent time loss. However, in mean square error, the loss is calculated
by  squared  the  difference  between  estimated  time  and  actual  time,
resulting in a simple linear relationship of time error across the entire
timeline.  However,  in  practice,  our  error  requirements  for  time  esti-
mation  are  non-linear.  For  example,  when  estimating  time  for  two
fingerprint images with real times of 10 s and 100 s, when the error is
both 1 s, it is obvious that we should assign larger errors to images with
smaller real times instead of giving the same error value. Therefore, it is
necessary to modify the mean square error function, and the time loss
after modification is:

InfraredPhysicsandTechnology138(2024)1052236X. Yu et al.

Ltime =

1
N

∑N

i=1

√
̅̅̅
(
ti

(cid:0)

√
̅̅̅̅
̂t i

)2

Performing a square root operation on the time in the original mean
square error can amplify the loss value when the time is small. As shown
in  Fig.  5,  the  spacing  between  t1t2t3  is  the  same,  and  the  difference
obtained under the same spacing for line y1 is the same. However, for the
y2  curve,  the  difference  obtained  between  t1t2  is  significantly  higher
than t2t3, which can amplify the loss.

Classifier 2 also goes through two fully connected layers for feature
extraction  and  final  multiclassification.  Since  the  infrared  handprint
image  has  the  problem  of  small  interclass  distance,  especially  the
interclass  distance  may  be  smaller  than  the  intraclass  distance  under
time blurring, it may be difficult to effectively classify the feature in-
formation by using the traditional softmax, so this paper chooses Center
Loss  [29]  to  supervise  the  learning  of  identity  features  for  identity
classification.

The center loss is a loss function used to increase the spacing between
classes, and its goal is to bring the feature vectors of samples of the same
class in the feature space closer to the center of the class, which results in
an  increase in the spacing  between classes. The loss function  form of
center loss consists of two parts, one part is the common Softmax cross-
entropy  loss,  which  is  used  to  ensure  that  the  model  can  classify
correctly.  The  other  part  is  the  center  loss  term,  which  is  used  to
constrain the distance of the sample’s feature vector from the center of
the category.

For the i-th sample, the Softmax cross-entropy loss and center loss

terms are:

⎛

⎝

L(i)
softmax = (cid:0)

log

(

exp

)

(cid:0)

x(i)

+ byi

)

⎞

⎠

)

W T
yi⋅f
(

C
j=1exp

W T

j ⋅f (x(i) ) + bj

∑

L(i)
center =

)

x(i)

(cid:0)

⃦
1
⃦f
2

⃦
⃦2

(cid:0) cyi

where  C  denotes  the  number  of  identity  categories,  W  and  b  are  the
weight and bias of the softmax layer, respectively, and yi  denotes the
(cid:0)
corresponding real identity label. f
denotes the feature vector of
the i-th sample. cyi  is the category center of the true category of the i-th
sample.  The  loss  of  the  final  classifier  2  is  the  weighted  sum  of  the
Softmax cross-entropy loss and the center loss term:

(i)

x

)

id = L(i)
L(i)

softmax + λL(i)

center

where λ is the weight of the central loss term, which is used to balance
the importance of the two loss terms.

Loss  of  center  reinforces  the  differences  between  categories  by
introducing  the  concept  of  category  centers, which  draws  the  feature
vectors of samples of the same category in the feature space closer to the
category centers. By minimizing the distance between the feature vec-
tors of the samples and the category centers of their true categories, the
center  loss  forces  the  model  to  learn  feature  representations  with
stronger differentiation, making the samples of the same category more
aggregated in the feature space and the samples of different categories
more dispersed, which effectively increases the spacing between cate-
gories and improves the model’s differentiation of different categories.
There are different gaps between the identity classes of infrared hand-
prints,  and  we  expect  Center  Loss  to  encourage  strong  discriminative
power in the learned features for trace identity recognition.

The loss of the final feature separation network is the weighted sum

of two individual losses, as shown in the formula:

Ltotal = λidLid + λtimeLtime =
λidlcosface(B(Xid), yid) + λtimelmse(A(Xtime), ytime)

Among them, the first item is Center Loss, and the second item is time
estimation loss. Xid  and Xtime  represent the identity component and time

component of feature decomposition, respectively. yid and ytime represent
the true identity category and departure time of the sample. λ Represents
a weight parameter that controls the balance of different loss terms. We
continuously  optimize  the  weight  parameters  as  trainable  parameters
during the training process.

4. Experiment

4.1. Dataset construction

For the study of handprint images, we employed the Fluke TIX640
infrared thermal imager, which has a resolution of 640 × 480, a spectral
range of 7.5 to 14 µm, and a thermal sensitivity of 0.03℃. During the
research process, we recruited 14 healthy volunteers, including 9 males
and 5 females, who had a healthy hand condition, had no hand injuries,
and were able to complete designated movements according to experi-
mental  requirements.  The  experiment  stipulated  four  sets  of  hand
movements,  including left  hand  finger convergence  (LFT),  right hand
finger convergence (RFT), left hand finger separation (LFA), and right
hand finger separation (RFA).

In order to ensure that infrared fingerprint images are not affected by
external  factors  such  as  device  automatic  compensation  and  environ-
mental  temperature  changes  during  data  collection,  we  conducted  a
unified and standardized collection process on 6 volunteers, including 5
males  and  1  female.  Each  volunteer  collects  the  four  sets  of  hand
movements  mentioned  above  and  forms  a  set  of  infrared  fingerprint
sequence images in chronological order, totaling 280 images. In order to
reduce  single  group  errors,  we  conducted  three  independent  image
acquisition processes.

After capturing all hand trace images, we performed data processing.
Firstly, by manually measuring the center position of hand marks, we
cropped each group of images into squares. To ensure sufficient spacing
between the boundary of the crop box and the contour of the hand, and
considering that thermal diffusion may cause the edge features of the
hand to diffuse outward over time, we retained a distance of more than
half a finger.

The  final  dataset  is  named  IRHTv2c6,  which  includes  the  main
standard  dataset  and  some  additional  subsets.  The  main  IRHTv2c6
dataset  collected  4913  images  from  6  subjects,  each  presenting  a
sequence of four types of hand tracking images. In order to conduct fair
and meaningful experimental comparisons, we divided the dataset into
training and testing sets. Two images of the same hand type were used as
the training set for each participant, while the remaining one was used
as the testing set. This partitioning method helps to verify the perfor-
mance and generalization ability of the model.

4.2. Experimental details

In the experiment of infrared handprint identity recognition and time
estimation,  we  improved  model  performance  by  employing  data
augmentation methods. Data augmentation primarily considered hand
pose  variables  and  data  cropping  errors,  including  random  trans-
formations of position, size, and rotation angles. The rotation method
involved  randomly selecting  rotation  values  between  (cid:0) 10  and  10  for
each  sample,  while  the  displacement  method  was  implemented  by
introducing random offsets of 0 % to 10 % along the x and y axes, along
with horizontal random flipping. The processing steps included align-
ment  to  224  × 224,  random  rotation,  random  horizontal  flipping,
random cropping to 200 × 200, and finally alignment to 112 × 112. The
data for the test set was directly aligned to 112 × 112.

For time estimation labels, in order to reduce the span of time labels,
we introduced a time scale factor set to 30 to prevent failures during the
training process.

During  the  training  process,  we  chose  stochastic  gradient  descent
(SGD)  as  the  optimizer  with  an  initial  learning  rate  of  0.0005  and  a
momentum  coefficient  of  0.9  to  accelerate  training.  To  reduce

InfraredPhysicsandTechnology138(2024)1052237X. Yu et al.

overfitting, we applied L2 regularization with a coefficient of 0.0005.
The model is trained using servers with NVIDIA GTX 2080Ti GPU. The
overall training is configured for 100 epochs, with each batch training
16  images.  The  initial  learning  rate  is  set  to  0.0005,  and  during  the
training  process,  the  learning  rate  is  reduced  to  80  %  of  the  original
value at the 30th, 60th, and 80th epochs. For the Center Loss function
used  in  identification,  the  weight  parameter  of  the  hyperparametric
center loss term is set to 0.35. To standardize the image size, input im-
ages are uniformly resized to 112 × 112, followed by linear normali-
zation to ensure pixel values are within the range of [-1, 1].

4.3. Separation of feature loss weight learning

A key challenge in training deep feature separation models is how to
balance the weights between feature losses under different tasks, so that
the model can achieve good performance in all feature extraction. Ac-
cording to the weight adjustment method proposed in reference [30,31],
we  present  the  process  of  weight  learning  optimization.  In  the  deep
feature  separation  model,  by  defining  a  total  loss function,  the  losses
generated  by  features  under  different  tasks  are  combined  and  equal
weights  are  initialized.  During  the  training  process,  the  total  loss  is
calculated through backpropagation, and model parameters are updated
based on the weight of each task. Regularly evaluate task performance
on the validation set and adjust weights based on performance, so that
better  performing  tasks  receive  higher  weights  to  achieve  dynamic
balance. For the initial weights, due to the different loss functions of the
identity recognition task and the time estimation task, when the same
initial weight is selected, the time estimation loss is much greater than
the identity recognition loss, which is not conducive to stable training of
the network and may even cause training failure. Therefore, scale the
time results to scale the loss. The final choice is to use the sgd optimizer
with timestamps processed by scale for deep feature separation model
learning.

4.4. Performance evaluation of identity prediction and time estimation

Identity prediction evaluation:
To  verify  the  performance  of  the  model,  we  conducted  a  compre-
hensive  performance  comparison  evaluation  between  the  Deep  Soft
Threshold Feature Separation (DSTFS) method and multiple models in
the infrared fingerprint identity recognition task. The compared model
consists of two parts. One part is the existing multi task training model,
including HF-ALEX [32], HF-RESNET [32], HE-CNN [32], and MTLface
[23]. The other part is a modified multi task model based on the classic
single  task  image  classification  model,  including  Resnet18  multi,
Efficientnet-b0 multi, and Regnet_X_400mf multi, etc. The experimental
results of each model are shown in Table 2. The Test Accuracy in the
table  represents  the  probability  of  the  model  predicting  the  correct
identity on the test set, which is the ratio of the number of predicted
correct identities to the number of all images in the entire test set.

From the overall results, HE-RESNET_CBAM and MTLface performed

better compared to other models, achieving accuracies of 74.29 % and
75.58  %,  respectively.  The  model  proposed  in  this  article  achieved  a
testing accuracy of 80.45 %, surpassing all other models. This indicates
that  our  method  has  excellent  performance  in  infrared  fingerprint
identity recognition tasks and can more accurately predict identity in-
formation.  In  contrast,  our  model  achieves  a  better  balance  between
parameter  quantity  and  accuracy,  with  relatively  small  parameter
quantity and significant improvement in accuracy.

Time estimation evaluation:
In the infrared fingerprint time estimation task, in order to gain a
deeper  understanding  of  the  performance  of  various  models  within
different error ranges, we conducted a detailed test set evaluation and
comprehensively compared their performance data. The selected model
for comparison was the same as the identity estimation experiment. In
the test results, we set the error rate of time estimation as the ratio of the
time estimate outside the allowable deviation to the total estimate. In
the experiment, we set 60 s and 120 s as the allowed deviation thresh-
olds, and the obtained experimental results are shown in Table 3.

The  Deep  Soft  Threshold  Feature  Separation  Model  (DSTFS)  per-
forms well in infrared fingerprint time estimation tasks. Under the two
indicators of Accuracy-60 and Accuracy-120, our model achieved 72.82
% and 94.29 %, respectively. Compared with other models, our algo-
rithm achieved relatively high accuracy results in both error ranges. This
indicates that the DSTFS model has significant advantages in accurately
estimating infrared fingerprint time. Furthermore, the model proposed
in this article has a relatively small standard deviation in Accuracy-60
and  Accuracy-120,  indicating its  high  stability  on different  data  sam-
ples. This may indicate that DSTFS has better generalization ability for
different scenarios and samples.

4.5. Ablation experiment

Multi task ablation experiment:
To evaluate the effectiveness of the deep soft threshold feature sep-
aration  model  compared  to  the  unseparated  feature  model  and  to
explore  the  superiority  of  feature  separation  in  infrared  fingerprint
identity  recognition  and  time  estimation  tasks,  we  designed  ablation
experiments  by  comparing  the  performance of  the  deep  feature  sepa-
ration model and the unseparated feature model in identity recognition
and time estimation tasks.

For our proposed deep soft threshold feature separation models for
identity  recognition  and  time  estimation,  we  design  separate  control
models for undivided feature networks under a single task. For identity
recognition  tasks,  the  network  part  related  to  time  estimation  in  the
feature separation model is removed, and only the identity recognition
branch network is retained. After removing the attention module, the
impact  of  time  estimation  features  on  identity  recognition  features  is
estimated. The output of the attention module of the identity recognition
branch is directly used as the feature output of the model for identity
recognition classification. Under the identity recognition task, the pre-
diction  loss  and  accuracy  results  of  feature  separation  and  non

Table 2
Identity prediction performance in the test set.

Table 3
The estimation performance of infrared fingerprint departure time.

Model

Params(M)

Test Accuracy(%)

Model

Accuracy-60(%)

Accuracy-120(%)

HF-ALEX
HF-RESNET18
HE-RESNET_CBAM
HF-RESNET34
HE-CNN
MTLface
Resnet18-multi
Efficientnet-b0-multi
GoogleNet-multi
Regnet_x_400mf-multi
DSTFS

6.43
24.56
24.59
59.75
25.59
50.43
23.01
46.50
6.44
13.76
52.96

69.81 ± 2.77
71.86 ± 3.90
74.29 ± 3.17
76.27 ± 2.29
71.79 ± 2.61
75.58 ± 2.40
75.96 ± 1.77
72.18 ± 2.26
68.33 ± 1.97
70.90 ± 1.79
80.45 ± 2.16

HF-ALEX
HF-RESNET18
HE-RESNET_CBAM
HF-RESNET34
HE-CNN
MTLface
Resnet18-multi
Efficientnet-b0-multi
GoogleNet-multi
Regnet-multi
DSTFS

67.63 ± 2.08
64.04 ± 2.46
70.38 ± 2.31
70.13 ± 4.53
69.12 ± 2.66
72.18 ± 1.49
71.97 ± 1.84
68.78 ± 2.32
70.13 ± 1.30
65.64 ± 2.03
72.82 ± 1.60

92.44 ± 1.27
91.86 ± 1.82
91.35 ± 2.34
92.88 ± 4.08
91.30 ± 2.49
90.47 ± 0.95
92.54 ± 1.40
91.86 ± 1.80
92.18 ± 1.00
89.55 ± 1.66
94.29 ± 1.44

InfraredPhysicsandTechnology138(2024)1052238X. Yu et al.

separation are shown in Fig. 8.

Compared to the unseparated feature model, our proposed feature
separation model has smaller loss and higher recognition accuracy in the
validation dataset. The introduction of time estimation tasks has had a
positive impact on network training.

For  individual  time  estimation  tasks,  the  identity  recognition
network part in the feature separation model is removed, and the overall
network  architecture  is  to  extract  features  through  multiple  residual
convolution blocks and further extract time features through multiple
attention mechanisms, which are applied to the regression task of time
estimation. The time estimation of unseparated features and the time
prediction  loss  and  error  results  of  separated  features  are  shown  in
Fig. 9.

From the graph, it can be seen that feature separation has a lower loss
compared to non separation time estimation, but the difference is not
significant. For the error results within 60 s and 120 s, the error results
within 60 s are also basically the same, but there is a certain improve-
ment in the error within 120 s. It is possible that in the structure of the
feature  separation  network  model,  only  the  temporal  features  are
superimposed on the identity features, and the identity features do not
affect the temporal features, resulting in similar time estimation results
with unseparated features in the feature separation mode.

However,  from  the  time  estimation  loss  curve  of  the  training  set
(Fig.  10),  it  can  be  seen  that  the  loss  curve  of  the  unseparated  time
estimation is relatively flat, while the loss curve of the feature separation
model shows two stages of decline, and the final loss is slightly lower
than that of the unseparated feature model. This situation may be due to
the impact of identity recognition errors under the feature separation
model, which ultimately exhibits positive effects.

From the three graphs above, it can be seen that the feature sepa-
ration  model  experiences  significant  fluctuations  in  both  loss  and  ac-
curacy in the early stages of model training. This may be due to the fact
that  learning  the  feature  separation  model  involves  balancing  and
competing between different feature weights, and the model may need

Fig. 8. Identity loss and recognition accuracy curve of feature separation and
non separation models.

Fig. 9. Curve of Time Estimation Loss and Error Rate for Feature Separation
and Unseparated Models.

Fig. 10. Loss Curve of Feature Separation and Unseparated Time Estimation in
the Training Set.

to  allocate  resources  and  learning  priorities  between  different  tasks,
resulting in overfitting on certain tasks and underfitting on other tasks,
This leads to significant fluctuations in the loss curve.
Comparison of Time Estimation Loss Functions:
To  verify  the  effectiveness  of  the  loss  function  designed  for  time
estimation in this article, a comparative experiment was conducted with
the  mean  square  error  loss  function  to  verify  the  performance  of  the
improved loss. The validation experiment selected a set of infrared fin-
gerprints with gradually increasing departure time for prediction, and
the experimental results are shown in Fig. 11.

The black solid line in the figure represents the actual departure time
of  the  infrared  fingerprint,  while  the  two  dashed  lines  represent  the
predicted time under mean square error loss and the predicted time after
improving  the  loss  function.  It  can  be  seen  that  the  improved  loss

InfraredPhysicsandTechnology138(2024)1052239X. Yu et al.

review  &  editing,  Visualization,  Project  administration  and  Software.
Baofeng  Zhang:  Funding  acqusition,  Investigation,  Resources.  Hao
Xue:  Data  curation,  Writing  –  review  &  editing,  Visualization,  Meth-
odology, Resources, Project administration, Software.

Declaration of competing interest

The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.

Data availability

Data will be made available on request.

Fig. 11. Comparison results of improved loss functions.

References

function makes the overall error of the model smaller, while also paying
more attention to errors in shorter time, thus making the model more
accurate in predicting in shorter time.

5. Conclusion

Heat trace recognition technology can predict the identity of targets
and estimate their departure time by analyzing the captured heat traces.
However,  due  to  the  influence  of  thermal  diffusion,  over  time,  the
captured thermal traces often exhibit a deep fuzzy state. In this situation,
extracting time invariant identity features from deep fuzzy heat traces
becomes crucial. Solving this challenge will help improve the accuracy
and reliability of heat trace recognition technology, thereby promoting
the development of criminal investigation. This article focuses on the
task  of  infrared  fingerprint  identity  recognition.  By  using  a  deep  soft
threshold feature separation method, identity information features and
time information features are separated, effectively improving the ac-
curacy of infrared fingerprint identity recognition and time estimation.
In  terms  of  network  architecture  design,  a  feature  separated  identity
recognition  and  time  estimation  network  is  proposed  based  on  the
ResNet architecture. Deep feature extraction is achieved through mul-
tiple  residual  convolutional  blocks,  and  spatial  and  channel  attention
mechanisms are introduced to handle the problem of feature separation.
To enhance the model’s expressive power, the soft threshold activation
function SPRelu was used to deal with noise in residual blocks.

Regarding the weight balance of deep feature separation, this article
achieves dynamic balance by defining a total loss function and period-
ically  evaluating  task  performance  on  the  validation  set.  In  order  to
solve the initial weight problem, the time estimation results were scaled
and  ultimately  the  sgd  optimizer  was  selected  for  multi  task  weight
learning.

In the experimental section, the performance evaluation of identity
prediction and time estimation was conducted on the proposed feature
separation  model,  and  the  effectiveness  of  the  method  was  verified
through  comparison  with  other  multitasking  models  and  classical
models.  In identity prediction tasks,  the proposed model significantly
outperforms other models in testing accuracy, while in time estimation
tasks, the model performs well with lower error rates and higher sta-
bility. The positive effects of multi task learning were further confirmed
through ablation experiments of multiple and single tasks.

CRediT authorship contribution statement

Xiao Yu: Conceptualization, Data curation, Writing – original draft,
Writing – review & editing, Validation, Formal analysis, Methodology,
Supervision.  Xiaojie  Liang:  Data  curation,  Writing  –  original  draft,
Writing-review &  editing, Methodology, Supervision, Project adminis-
tration  and  Software.  Zijie  Zhou:  Writing  –  original  draft,  Writing  –

[1] S. Bitzer, O. Ribaux, N. Albertini, et al., To analyse a trace or not? Evaluating the
decision-making process in the criminal investigation, Forensic Sci. Int. 262 (2016)
1–10.

[2] Z. Zhou, B. Zhang, X. Yu, Immune coordination deep network for hand heat trace

extraction, Infrared Phys. Technol. 127 (2022) 104400.

[3] C.H. Edlin, Infra-Red Rays in Criminal Investigation, Police J. 8 (4) (1935)

458–466.

[4] A. Rogalski, Recent progress in infrared detector technologies, Infrared Phys.

Technol. 54 (3) (2011) 136–154.

[5] S. Wang, Y. Du, S. Zhao, et al., Multi-scale infrared military target detection based

on 3X-FPN feature fusion network, IEEE Access (2023).

[6] B.B. Lahiri, S. Bagavathiappan, T. Jayakumar, et al., Medical applications of

infrared thermography: a review, Infrared Phys. Technol. 55 (4) (2012) 221–235.
[7] D. Li, X.P. Zhang, M. Hu, et al., Physical password breaking via thermal sequence

analysis, IEEE Trans. Inf. Forensics Secur. 14 (5) (2018) 1142–1154.

[8] S. Naval, A. Pandey, S. Gupta, et al., PIN inference attack: A threat to mobile
security and smartphone-controlled robots, IEEE Sens. J. 22 (18) (2021)
17475–17482.

[9] K.K. Park, H.W. Suk, H. Hwang, et al., A functional analysis of deception detection

of a mock crime using infrared thermal imaging and the Concealed Information
Test, Front. Hum. Neurosci. 7 (2013) 70.

[10] A.Y. Fedorova, M.V. Bannikov, A.I. Terekhina, et al., Heat dissipation energy under
fatigue based on infrared data processing, Quantitative Infrared Thermogr. J. 11
(1) (2014) 2–9.

[11] A. Krizhevsky, I. Sutskever, G.E. Hinton, Imagenet classification with deep

convolutional neural networks, Adv. Neural Inf. Proces. Syst. 25 (2012).
[12] K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale

image recognition. arXiv preprint arXiv:1409.1556, 2014.

[13] K. He, X. Zhang, S. Ren et al., Deep residual learning for image recognition, in:
Proceedings of the IEEE conference on computer vision and pattern recognition,
2016, pp. 770-778.

[14] M. Tan, Q. Le, Efficientnet: Rethinking model scaling for convolutional neural

networks[C]//International conference on machine learning, PMLR (2019)
6105–6114.

[15] I. Radosavovic, R.P. Kosaraju, R. Girshick et al., Designing Network Design Spaces,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2020, pp. 10428–10436.

[16] C. Szegedy, W. Liu, Y. Jia et al., Going Deeper with Convolutions, in: Proceedings

of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp.
1–9.

[17] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., An Image is Worth 16x16 Words

Transformers for Image Recognition at Scale, in: International Conference on
Learning Representations, 2021.

[18] I. Jancskar, A. Ivanyi, Fuzzy-rule based diffusion in thermal image processing,

Pollack Periodica 1 (1) (2006) 115–129.

[19] W. Chen, Determination of displacement from an image sequence based on time-
reversal invariance, IEEE Trans. Geosci. Remote Sens. 52 (5) (2013) 2575–2592.
[20] A. Suzuki, T. Hoshino, K. Shigemasu, et al., Decline or improvement?: Age-related

differences in facial expression recognition, Biol. Psychol. 74 (1) (2007) 75–84.
[21] R.R. Atallah, A. Kamsin, M.A. Ismail, et al., Face recognition and age estimation
implications of changes in facial features: A critical review study, IEEE Access 6
(2018) 28290–28304.

[22] C. Wu, H.J. Lee, Learning age semantic factor to enhance group-based

representations for cross-age face recognition, Neural Comput. & Applic. (2022)
1–12.

[23] Z. Huang, J. Zhang, H. Shan, When Age-Invariant Face Recognition Meets Face Age

Synthesis: A Multi-Task Learning Framework, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2021, pp. 7282–7291.
[24] J. Hu, L. Shen, G. Sun, Squeeze-and-Excitation Networks, in: Proceedings of the

IEEE Conference on Computer Vision and Pattern Recognition, 2018, 7132–7141.

[25] M. Jaderberg, K. Simonyan, A. Zisserman, Spatial transformer networks, Adv.

Neural Inf. Proces. Syst. 28 (2015).

InfraredPhysicsandTechnology138(2024)10522310X. Yu et al.

[26] S. Woo, J. Park, J.Y. Lee et al., Cbam: Convolutional Block Attention Module, in:
Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp.
3–19.

[27] D.L. Donoho, De-noising by soft-thresholding, IEEE Trans. Inf. Theory 41 (3)

(1995) 613–627.

[28] H. Liu, B.R. Foygel, Between hard and soft thresholding: optimal iterative
thresholding algorithms, Inform. Inference: J. IMA 9 (4) (2020) 899–933.
[29] Y. Wu, H. Liu, J. Li, et al., Improving face representation learning with center

invariant loss, Image Vis. Comput. 79 (2018) 123–132.

[30] B. Lin, Y.E. Feiyang, Y. Zhang, A closer look at loss weighting in multi-task

learning, 2021.

[31] T. Gong, T. Lee, C. Stephenson, et al., A comparison of loss weighting strategies for

multi task learning in deep neural networks, IEEE Access 7 (2019)
141627–141632.

[32] R. Ranjan, V.M. Patel, R. Chellappa, Hyperface: A deep multi-task learning

framework for face detection, landmark localization, pose estimation, and gender
recognition, IEEE Trans. Pattern Anal. Mach. Intell. 41 (1) (2017) 121–135.

InfraredPhysicsandTechnology138(2024)10522311