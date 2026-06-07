Contents lists available at ScienceDirect

Expert Systems With Applications

journal homepage: www.elsevier.com/locate/eswa

Multi-task learning for hand heat trace time estimation and
identity recognition

Xiao Yu a, 1, Xiaojie Liang a, Zijie Zhou b, *, 2, Baofeng Zhang a
a School of Electrical Engineering and Automation, Tianjin University of Technology, Tianjin, China
b Ningbo Institute of Dalian University of Technology, Zhejiang, China

A R T I C L E  I N F O

A B S T R A C T

Keywords:
Heat traces recognition
Thermal imaging
Multi-task learning
Deep neural network

The  thermal  imager  can  capture  the  invisible  thermal  trace,  and  the  potential  information  contained  in  the
thermal trace can be extracted through the identification and analysis of the infrared thermal trace, such as the
identity information and time information of the trace. This technology has great application value in criminal
investigation, military and other fields. However, with the passage of time, the heat traces will gradually fade,
and  the  intra-class  distances  will  gradually  increase.  For  hand  heat  trace,  the  trace  may  exhibit  significant
variations  at  different  time  points  or  in  different  states.  This  will  pose  challenges  for  extracting  potential  in-
formation and present significant challenges for model recognition. Therefore, we propose a multi task frame-
work  using  deep  convolutional  neural  networks  (MTLHand)  to  jointly  handle  the  two  tasks  of  heat  trace
identification and heat trace departure time estimation. It can learn invariant representations of thermal trace
identities, better capturing the dynamic changes in palmprint data, to enhance the accuracy of identity recog-
nition.  Specifically,  soft  threshold  is  inserted  as  nonlinear  transformation  layers  into  an  improved  ResNet  to
extract deep mixed hand trace features. The mixed features are decomposed into identity related features and
time related features through spatial attention and channel attention. Use the multi task training method to carry
out  the  corresponding  learning  tasks.  In  addition,  we  collected  and  constructed  an  infrared  hand  heat  trace
dataset containing labels such as departure time, gender, hand posture, identity category, etc., to promote the
development of heat trace research. Experiments on this dataset show that our method is superior to other deep
learning  methods  for  the  specific  task  of  infrared  heat  trace  recognition,  and  the  identification  accuracy  can
reach 83.48 %. In the trace leaving time estimation task, the recognition error rate is 18 % if the residual value
between the estimated time and the real time is less than 60 s, and 3.55 % if the residual value is less than 120 s.

1. Introduction

1.1. Background

The development of science and technology has two sides to social
security and stability. On the one hand, the police can comprehensively
track criminals and improve the efficiency of case detection by relying
on universal monitoring equipment. On the other hand, criminal means
have  become diversified  increasingly,  and  anti-reconnaissance  aware-
ness has become stronger. They will consciously prevent their biological
information  from  being  left  behind,  such  as  wearing  headgear  and

gloves,  and  have  a  comprehensive  method  to  fight  against  police
investigation. The promotion of criminal ability and the popularization
of  high-tech  bring  new  challenges to  police criminal  investigation. In
this context, the research on criminal investigation by scholars in rele-
vant fields has never stopped, such as forensic DNA analysis technology
(Lavergne  et  al.,  2022;  Szkuta  et  al.,  2019),  trace  material  evidence
identification technology (Fitch & Touroo, 2018). Although the existing
technologies  are  constantly  improving  and  perfecting,  they  all  have
different technical characteristics. DNA analysis technology utilizes the
DNA to  identify criminals  with high matching  degree  and sensitivity.
However,  the  detection  process  is  cumbersome,  which  will  delay  the

* Corresponding author.

E-mail  addresses:  yx_tjut@email.tjut.edu.cn  (X.  Yu),  lxj437294988@stud.tjut.edu.cn  (X.  Liang),  zhouzj_nbi@dlut.edu.cn  (Z.  Zhou),  zhangbaofeng@tjut.edu.cn

(B. Zhang).

1  ORCID: 0000-0001-5762-2503.
2  ORCID: 0009-0004-3735-8564.

https://doi.org/10.1016/j.eswa.2024.124551
Received 26 April 2023; Received in revised form 18 May 2024; Accepted 18 June 2024

ExpertSystemsWithApplications255(2024)124551Availableonline21June20240957-4174/©2024ElsevierLtd.Allrightsarereserved,includingthosefortextanddatamining,AItraining,andsimilartechnologies.X. Yu et al.

opportunity  of  investigation  to  a  certain  extent.  The  effectiveness  of
trace material evidence identification technology largely depends on the
integrity of the fragmented evidence collected at the crime scene. More
importantly, these technologies that are well known to the public are
also concerned by criminals, who will consciously fight to prevent being
caught.

The application of thermal imaging technology in criminal investi-
gation  can  complement  the  above  technologies.  Thermal  imaging
technology has been widely used in various fields, such as biomedical
engineering  (Scebba  et  al.,  2021;  Hu  et  al.,  2018;  Majumdar  et  al.,
2019),  intelligent  transportation  (Chen  et  al.,  2022;  Hu  et  al.,  2021),
intelligent buildings (Li and Chidurala, 2021; Li and Chidurala, 2022)
and  security  industry  (Maimunah  Mohd  Ali  et  al.,  2020).  Due  to  the
nature of the thermal imager, when the object is touched by the human
body, it can capture the residual human heat trace information on the
object. Through the heat trace information, we can infer the time when
the human body left and the behavior habits.

Thermal imaging technology is a passive imaging technology (Hou
et al., 2022; Buddharaju et al., 2007). Thermal imaging technology can
quickly capture and track heat trace information. Under certain condi-
tions, it can play its technical advantages to help solve difficult cases. At
the crime scene, even if criminals actively prevent leaving hair, textile
fibers and other clues that can be found by human eyes, they cannot
avoid  leaving  heat  traces  with  biological  information  at  the  scene.
Because  criminals  often  do  not  carry  professional  trace  detection
equipment  when  engaging  in  criminal  activities,  they  cannot  directly
observe the remaining thermal traces. When destroying on-site traces, it
is inevitable to overlook the thermal trace evidence related to biological
information that can be captured by thermal imaging technology. The
information  of  hand  trace  can  be  obtained  by  using  thermal  imaging
technology  to  capture  the  image  of  the  criminal’s  hand  contact.  Ac-
cording  to  the  analysis  of  trace  characteristics,  the  identity  of  the
perpetrator can be predicted and the time of departure can be estimated.
Of  course,  any  task  of  trace  correlation  recognition  through  infrared
imaging technology will be subject to time constraints. After the crime,
as time goes on, when the residual heat of the trace gradually reaches the
ambient temperature, this technology will fail to some extent. Therefore,
if we can obtain the residual heat trace at the scene within a certain time
window after the perpetrator leaves the crime scene, it may possibly be
of great help to the detection.

The  existing  relevant  literature  has  the  work  of  cracking  codes
through the residual heat left by hands. To the best of our knowledge,
none of the reported technologies used thermal imaging technology and
deep learning to simultaneously identify hand heat traces and estimate
time. Hand heat trace is a kind of criminal trace that is easy to be ignored
and difficult to be quickly eliminated. Within definite time window, the
police  can  use  the  heat  trace  left  at  the  crime  scene  to  capture  the
relevant information of the crime. Heat trace can be used to infer the
target and crime mode of the criminal, which is of great significance to
narrow  the  search  scope  of  suspect.  Therefore,  we  propose  a  time
invariant infrared hand trace recognition method. Through the research
on the infrared hand heat trace model, we extract the identity related
information and time related information from the heat trace image, and
combine them to improve the identity recognition performance of the
hand trace and estimate the leaving time of the trace.

1.2. Aims and objectives

The aim of this work is to study a kind of hand heat trace recognition
model. This model can effectively identify the identity category corre-
sponding  to  the  hand  trace  in  the  trace  investigation  scene  of  the
criminal investigation scene based on the hand trace biological infor-
mation, and estimate the departure time of the target.

We  designed  and  evaluated  a  deep  learning  model  to  effectively
extract and express the identity information and time information in the
hand heat trace dataset we built (detailed composition is in section 3).

We compared the performance of the current advanced deep learning
model, and  found that the test  accuracy of identity classification will
generally  converge  to  a  certain  interval,  and  the  change  of  model
structure  cannot  improve  the  performance.  Therefore,  multi  task
learning is selected to learn identity recognition task and time estima-
tion task at the same time, so that the two tasks can promote each other
and improve the model performance.

One  of  the  main  challenges  facing this  task  is  the  issue  of  feature
changes  over  time.  As  time  passes,  infrared  handprint  images  of  in-
dividuals gradually blur and dissipate due to thermal diffusion, resulting
in significant differences in handprints of the same person over different
time periods. This inherent change in intra-class distance poses a sig-
nificant  challenge  for  traditional  handprint  recognition  systems.  Our
goal is to address the problem of intra-class distance variation over time
in  handprint  recognition,  in  order  to  improve  the  stability,  accuracy,
reliability,  and  practicality  of  the  system.  By  effectively  reducing  the
fluctuation of intra-class distance, the system can accurately recognize
handprints over long time spans without being affected by time factors.
Ultimately,  through  continuous  monitoring  and  adjustment  of  intra-
class  distance,  we  aim  to  dynamically  optimize  the  system  to  ensure
high  levels  of  performance  and  effectiveness  in  constantly  changing
environments.

We analyze the characteristics of the identity information and time
information  of  heat  trace,  find  the  recognition  targets  with  similar
characteristics  to  the  hand  heat  trace,  and  use  the  corresponding
recognition model for reference and selection as the basic model of this
study.  According  to  the  interference  characteristics  of  heat  trace
recognition,  such  as  more  noise,  we  combine  the  soft  thresholding
module with the basic model to improve the model performance, and
compare  the  performance  of  the  proposed  method  with  the  most
advanced method.

Finally, we compare the identification performance of the heat trace
recognition model, and analyze the residual of the departure time esti-
mation results. The purpose is to understand the limitations of the deep
learning model based on hand trace data for trace targets with different
departure times and the performance characteristics of the learned time
estimation performance.

The paper is organized as follows: Section 2 presents the previous
research work. The details of the infrared hand trace database built is
described in Section 3. The method of hand heat trace identification is
presented in Section 4. The experiments are described in Section 5. The
analysis and discussion are presented in Section 6. Finally, the conclu-
sions of this research are given in Section 7.

2. Related work

2.1. Heat trace

The human body will leave heat traces on the surface of the object
when contacting the object, which can be clearly captured by the ther-
mal imager. This phenomenon can be used to record the user’s touch
trace  on  the  keyboard  to  crack  the  password  of  ATM  machines  and
electronic  lock safes.  Mowery et  al.  (2011)  described  the  principle of
password cracking technology based on thermal imaging, and proposed
to use thermal imaging technology to crack passwords. Hu et al. (2019)
established  a  physics  model  by  using  the  heat  sequence  images  that
could be  used to  crack  the  password, and  then  cracked the  password
through the inversion algorithm, with an accuracy of 26 %. Abdelrah-
man et al. (2017) studied the feasibility of cracking the unlock password
and mobile phone unlock mode by using the heat tracking technology.
The research results showed that the cracking accuracy was higher than
72 %. The similarity between these studies and our study lies in that the
realization  is  to  estimate  departure  time  of  heat  trace  or  judge  the
sequence of trace generation.

These researches utilized thermal imaging technology for password
cracking, although the research object is the same as our research, which

ExpertSystemsWithApplications255(2024)1245512X. Yu et al.

Table 1
Performance of classical models on identity recognition.

Model

Params (M)

Top-1 error (%)

Alexnet (Hinton et al., 2017)
VGG16 (Simonyan & Zissenman, 2014)
Resnet34 (He et al., 2016)
Resnet18 (He et al., 2016)
Efficientnet-b0 (Le & Tan, 2019)
Efficientnet-b7 (Le & Tan, 2019)
Regnet_x_400mf (Girshick et al., 2020)

57.01
134.28
21.28
11.17
4.01
63.80
5.10

38.53
31.54
29.94
29.10
27.63
32.63
31.09

Fig. 1. The major challenge of Hand heat trace recognition: the large intra-class
variations due to thermal diffusion.

is hand heat traces. But the key point of these studies is to distinguish the
sequence of trace generation. And we study the identification of hand
heat traces and estimate the departure time of hand traces in long time
window. In addition, some scholars also studied the application of foot
heat trace. Hu et al. (2020) established a thermal footprint model based
on Newton’s Law of Cooling to estimate the departure time. Ai et al.
(2020) developed a two-dimensional model to simulate human thermal
residues.  The  simulation  model  could  effectively  simulate  the  heat
dissipation process under different contact materials and contact time in
the real scene. These scholars focus on the application of thermal foot-
print in criminal investigation scene, and the research purpose is like to
our  study.  They  utilized  physics  models  to  estimate  the  time  by
comparing the heat trace sequence captured at the case scene with the
prior heat trace change. In our study, we used the deep learning method
to extract the hand trace features through training dataset, which can
predict the identity and estimate the departure time only through the
heat trace images. This method can only use the heat trace images taken
at  the  crime  scene  to  predict  the  identity  and  estimate  the  departure
time.

2.2. Hand heat trace identification

Significant progress has been made in the research and application of
heat tracing, but currently there is relatively little research in the field of
hand heat tracing recognition. The current method of identity recogni-
tion through hand thermal traces mainly focuses on infrared palm vein
biometric recognition. Chantaf et al. used convolutional neural networks
to classify and recognize palm vein images (Chantaf et al., 2020), which
can achieve contactless individual authentication with high security and
accuracy.

In addition, identity recognition can also be achieved by detecting
residual  infrared  thermal  traces  on  the  palm  of  the  hand.  We  have
proposed a deep network method based on biological immune coordi-
nation mechanism for extracting and tracking hand heat marks (Zhou
et al., 2022), which combines the feature extraction ability of the im-
mune  system  and  the  learning  ability  of  neural  networks  to  more

accurately extract fuzzy deep and irregularly shaped heat mark targets.
To achieve fuzzy classification of infrared hand heat marks, in previous
research, we used deep neural networks for hand mark identity classi-
fication (Zhou et al., 2021). Using an improved CNN architecture and a
small  MBConv  block  network  design,  we  improved  the  accuracy  and
performance  of  infrared  hand  heat  mark  classification.  We  train  an
identity recognition model using clear hand trace images from a set of
trace sequences, and test the model’s identity recognition performance
on  fuzzy  trace  images  in  the  same  set  of  sequences.  Under  specific
conditions, ideal identity recognition accuracy can be achieved. How-
ever, the sample size of the study is relatively small, and the training and
testing samples are from the same sequence, which does not meet the
practical application requirements.

In this study, a complete dataset is prepared for the recognition task.
We  constructed  an  infrared  hand  trace  dataset  containing  6  identity
categories, each of which contains 4 groups of different hand types, and
be repeatedly collected for 3 times. A total of 72 groups of hand trace
image sequences (4913 images) are used for recognition research. The
specific settings and division of dataset are described in section 3. We
compared  the  recognition  performance  of  networks  with  different
depths based on this dataset. All models are selected from the classic
classification models in the pytorch framework. The results are shown in
Table 1.

We  selected  48  groups  of  hand  trace  images  for  training  and  24
groups for testing. From the test results, we found that even if the model
finally converged in the training, it was still unable to obtain an ideal
identification accuracy in the test. We suspect that hand traces contain
other features that interfere with identity recognition, resulting in poor
performance. Due to the influence of thermal diffusion, the hand heat
trace  will  gradually  become  blurred  with  time.  It  has  been  shown  in
many literatures that the fuzzy characteristics of heat trace can affect the
recognition  and  extraction  of  hand  heat  trace.  Therefore,  we  further
investigate the impact of fuzzy features on identity recognition.

2.3. Depth blurring and time invariance of heat trace

The depth fuzzy feature of infrared hand traces has nearly always
been an influential factor in hand trace related tasks. To the best of our
knowledge,  there  are  few  literatures  that  directly  analyze  the  depth
fuzzy features in hand heat trace recognition, but there are studies on the
impact of target mixed features on the performance of recognition tasks,
such as cross age face recognition (Lee & Wu, 2022; Hu et al., 2017; Hu
et  al.,  2018).  Many  literatures  show  that  human  faces  will  produce
wrinkles  and  other  features  during  aging,  resulting  in  an  increase  in
intra-class distance and significant intra-class differences. This kind of
problem is called age-invariant face recognition problem. We find that
there are great similarities between age-invariant face recognition and
our research. Hand heat trace is a kind of biological feature, which can
indicate biological identity. As a result of thermal diffusion, the hand
heat trace will gradually blur over time, resulting in an intra-class dis-
tance change. It is often that the inter-class variation is much smaller
than the intra-class variation in the presence of time variation (d4 < d1),
as  illustrated  in  Fig.  1.  And  traces  of  the  same  identity  obtained  at
different time points will change more due to thermal diffusion, which is
more abstract and unobservable than faces with different ages. This kind
of depth blur shares similar time related information in all hand heat
traces. In this study, we call it time-invariant heat trace recognition.

Multi task learning has shown excellent performance in research that
includes  multiple  features  and  certain  features  change  with  specific
factors, such as the decomposition of identity and time features in hand
heat traces. There are two types of feature decomposition corresponding
to  different  tasks.  One  is  to  directly  classify  fusion  features  corre-
sponding  to  different  tasks.  For  example,  literature  (Chellappa  et  al.,
2017)  utilized  separate  CNN  to  fuse  the  middle  layer  of  deep  CNN,
connecting fusion features to independent full connection layers corre-
sponding to different tasks. The other is to obtain the features of one of

ExpertSystemsWithApplications255(2024)1245513X. Yu et al.

the impact of contact pressure on the result is not considered. And the
subjects are reminded in advance to naturally contact their hands with
the wall, so that they do not need to push the contact surface too much to
leave a clear trace. To avoid the influence of other heat sources in the
experimental environment, place the thermal imager at a fixed position
for image acquisition, as shown in Fig. 2. The thermal imager is placed
on a table 70 cm from the ground and 100 cm in front of wall. As the
thermal imager will generate heat after being used for a long time, it will
lead to problems affecting the trace imaging effect in the thermal trace
image,  as  shown  in  Fig.  3a.  We  set  the  thermal  imager  to  perform
background temperature compensation every 20 s, and select 10 s after
compensation  for  each  shooting  time.  In  addition,  by  controlling  the
shooting distance, the hand trace is captured with the size of the blue
box as shown in Fig. 3b to reduce the impact of thermal interference on
imaging.

3.2. Subjects

Total of 14 healthy subjects (9 males and 5 females) are selected. All
hands are uninjured and can complete all required actions. Each one is
informed that the experiment will be utilized for the scientific research
and not for any commercial purposes. All people are required to com-
plete the collection of four types of hand heat trace as shown in Fig. 4,
including left hand finger together (LFT), left hand finger apart (LFA),
right hand finger together (RFT), and right hand finger apart (RFA). In
the  experiment,  we  adopted  a  standardized  acquisition  process  for  6
subjects (5 men and 1 woman), so that there would be no obvious im-
aging problems due to the automatic correction of the equipment for the
hand  heat  trace.  For  the  six  subjects,  we  divided  the  trace  sequence
images  of  different  hand  types  into  one  set  (about  280  images)  and
collected three sets in total to prevent single group error. This dataset is
the main dataset we used in this study and is named IRHTv2c6 dataset.

3.3. Protocol

The  IRHTv2c6  dataset  records  four  different  types  of  hand  trace
images for each subject. The detailed acquisition process of trace images
of each type of hand is as follows:

First, subjects are given introduction to the experimental procedure,
the strength of the hand, the time of contact with the wall, and how to
make the correct type of hand trace.

Second,  initialize  the  thermal  imager  so  that  its  imaging  center  is
aligned with the wall mark. Perform autofocus on the wall without any
heat source interference.

Third,  inform  the  subjects  of  the  type  of  hand  they  needed  to
perform.  Subjects  are  required  to  place  their  hands  close  to  the  wall
without touching it. Then, focus the thermal imager manually until the
contour of the subject’s hand can be clearly displayed. After confirming
that  the  posture  and  position  of  the  hand  of  the  subject  meet  the  re-
quirements, the subjects are informed to fit the hand with the wall.

Fourth, when the subject’s hand is attached to the wall for 20 s, the
subjects are told to leave and a heat trace image is quickly collected after
leaving. After that, the heat trace images are collected within 10 s after
the  thermal  imager  performs  background  temperature  compensation.
The time interval between two adjacent acquisitions is about 5 s, with an
average of about 6 heat trace images collected per minute. When the
hand of the subject leaves, we ask the subject to leave naturally. During
this process, it is necessary to avoid generating strong air flow to prevent
impact on heat trace.

Fifth, the acquisition process of each group of hand trace images is
strictly implemented according to the above steps. After collecting the
hand  trace  data,  Fluke  Connetct  Smartview  software  is  used  for  data
processing.  The  specific  steps  include  setting  the  color  palette  of  the
infrared trace image to gray, recording the time information generated
by the image, and saving the image in jpg format. The labels of all trace
images are recorded in csv. For each hand trace image, we record the

Fig. 2. Acquisition environment.

Fig. 3. Acquisition details (a) thermal interference; (b) Location and size of the
captured trace.

the  tasks  by  processing  the  depth  mixed  features.  Then  subtract  the
original  feature  from  it  to  obtain  another  task  feature  (Huang  et  al.,
2021). In the experiment of this study, we tried two kinds of multi task
learning structures. We found that the second structure of decomposi-
tion depth mixed features is more suitable for time-invariant hand heat
trace recognition.

3. Hand heat trace dataset

Hand Trace Dataset (IRHT) is the basis of hand trace recognition. We
use FLUKE infrared thermal imager to shoot and collect hand traces. In
this  section,  the  details  of  heat  trace  acquisition  and  processing  are
described in detail.

3.1. Infrared thermal imager setup

A  fluke  tix640  infrared  thermal  imager  is  employed  to  record
infrared  hand  trace  images  with  resolution  640  × 480.  The  spectral
range of the thermal imager is 7.5 to 14 µm, and the thermal sensitivity
is 0.03
C,
and the measurement accuracy is 1.5 %.

C. The temperature measurement range is (cid:0) 40

C to 1200

◦

◦

◦

During the experiment, the subjects are required to fit their hands
with the contact target (wall) for 10 s to 30 s. Due to technical reasons,

Fig. 4. Four types of hand heat trace at different time.

ExpertSystemsWithApplications255(2024)1245514X. Yu et al.

Table 2
Statistical table for infrared handwriting image dataset.

Dataset name: IRHTv2c6

Number of subjects: 6

Subject

Hand shape

LFT

Dataset type: Hand heat tracking image

Number of images: 4913

LFA

RFT

RFA

0

1

2

3

4

5

Table 3
Partial dataset list.

Id

Image

Time

Left (0)/Right (1)

Type

Gender

ctime
(1)

ctime
(2)

0
1
3
0
4
4
4
3
5
0
0

AK081962
AG082337
AA081902
AP081906
AH082203
AF082153
AE082134
AC082159
AB090263
AO081903
AO081937

590
389
5
33
14
595
392
408
585
13
334

0
0
1
0
0
0
1
0
1
0
0

0
1
0
1
1
0
1
1
0
0
0

1
0
1
1
1
1
1
1
1
1
1

10
7
1
1
1
10
7
7
10
1
6

5
4
1
1
1
5
4
4
5
1
3

time0

60.39
52.39
1.33
58.49
58.59
42.19
38.51
40.41
16.22
24.07
29.28

identity, file name, departure time, subject gender, and hand trace type.
We set the departure time of the first image of the trace sequence to 1 s,
and  the  subsequent  departure  time  is  calculated  by  subtracting  the
corresponding  generation  time  from  the  generation  time  of  the  first
image.

3.4. Design of database

After  capturing  all  the  images,  each  group  of  images  is  cut  into
square by manually determining the center position of hand trace. The
distance  between  the  edge  of  the  clipping  box  and  the  hand  contour
should be more than half a finger (as shown in Fig. 3b), because the heat
diffusion will make the hand edge feature shift outward with time. Each
group  of  trace  images  forms  an  image  sequence  according  to  the
acquisition time. The relative position between the thermal imager and

the trace target does not change during the acquisition process to ensure
that the whole group of traces can be cut in batches based on the cutting
box of the first sequence. In summary, the IRHT database includes the
IRHTv2c6  standard  dataset  and  some  additional  subsets.  Its  main
IRHTv2c6 dataset contains 4913 images from 6 subjects. Each subject
displayed four types of hand trace sequence images. Each subject needs
to collect the same hand type three times. We divide the database into
two  parts:  training  set  and  test  set.  Taking  a  group  of  hand  trace
sequence images as the basic unit, take two of the same hand type of
each subject as the training set, and the remaining one as the test set to
make a fair and meaningful experimental comparison.

The final complete dataset information and statistics are shown in

Table 2.

The  dataset  contains  various  information  about  different  samples.
Each  row  of  data  corresponds  to  the  observation  results  of  a  specific

ExpertSystemsWithApplications255(2024)1245515X. Yu et al.

Fig. 5. An overview of the proposed MTLHand including two tasks.

Fig. 6. Three types of residual blocks in MTLHand. (a) represents a residual block with a stride of 2. The number of channels in the input feature map is the same as
that in the output feature map. (b) represents a residual block with a stride of 2. The number of channels in the input feature map is different from that in the output
feature map. (c) represents a residual block with a stride of 1.

sample at different time points. Among them, the id column represents
the identity identifier corresponding to the sample, the image column
represents the name of the sample image, the time column represents the
shooting  time  of  the  sample  since  generation,  the  left/right  column
represents whether the sample is left or right, the type column represents
the type of hand action, the gender column represents the gender of the
sample  source,  the  ctime  (1)  and  ctime  (2)  columns  are  time  related
stage markers, and the time0 column represents the specific recording
time point. The partial dataset list is shown in Table 3.

4. Methodology

4.1. Overview

The deep learning model used in this work is based on the resnet,

which has achieved excellent performance in multi task learning. Fig. 5
shows  the  structure  of  the  model.  We  use  a  resnet-like  network  for
feature  extraction,  which  consists  of  a  convolution  block  containing
convolution  layer,  BN  layer  and  activation  function  and  four  stages
containing several residual blocks. We decompose the extracted features
into identity related features and time related features, and classify and
regress  the  corresponding  tasks  through  the  fully  connected  layer
respectively. As described in Section 2, there are depth blurring features
caused by thermal diffusion in hand traces, which makes the identifi-
cation performance of hand traces unsatisfactory. Therefore, we use the
following perspectives to build our network:

1)  Hand  heat  trace  is  a  kind  of  biological  feature.  It  is  necessary  to
realize  the  identification  of  hand  traces  and  the  estimation  of  de-
parture  time  in  practical  applications.  Time  related  features  and

ExpertSystemsWithApplications255(2024)1245516X. Yu et al.

Fig. 7. Channel attention and spatial attention structure.

Since  the  input  image  resolution  is  112  × 112,  we  remove  the
pooling  layer  in  the  first  convolution  block  of  the  original  model.
Considering  that  the  convolution  operation  in  each  stage  is  prone  to
redundancy, we add the module based on soft thresholding to the res-
block with stride of 1 in the stage. Through the combination of channel
attention and spatial attention, the mixed features extracted after four
stages  are  decomposed  to  obtain  identity  related  features  and  time
related features, with the size of 7 × 7 × 512. Then, the identity related
features  and  time  related  features  are  stretched  into  one-dimensional
feature  vectors,  which  are  respectively  sent  into  the  fully  connected
layer with 512 neurons. Finally, add a fully connected layer to predict
the identity and estimate the time.

After each 3 × 3 convolution layer, we deploy the prelu activation
function. And rectified linear units (ReLU) is deployed after each fully
connected layer. We use specific loss functions for different tasks to learn
the weights of the network, and dynamically learn the loss allocation
weights of each task.

The proposed model implements different types of learning processes
within  a  single  framework  as  follows:  the  model  adopts  a  multi-task
learning  approach  with  hard  parameter  sharing,  where  the  bottom-
level features learned in the model are shared for extracting low-level
features from hand thermal traces. Simultaneously, the parameters of
the shared bottom feature network learn universal low-level features for
identity  prediction  tasks  and  time  estimation  tasks.  Following  this,  a
feature decomposition method is employed, utilizing residual mapping
to  decompose  the  feature  vector  into  two  unrelated  components.  An
attention  mechanism  is  used  to  supervise  the  learning  of  multiple
components, decomposing mixed feature maps in a high-level semantic
space.  Task-specific  loss  functions  are  introduced  on  top  of  hard
parameter  sharing,  employing  a  large-margin  cosine  loss  function  to
supervise identity feature learning for identity category prediction. The
Mean Squared Error (MSE) loss function is chosen for time estimation,
and the loss function is adjusted by measuring the co-variance uncer-
tainty  of  tasks,  enabling  the  model  to  simultaneously  learn  identity
prediction classification tasks and time estimation regression tasks with
different magnitude levels to better adapt to the requirements of each
task. Subsequently, independent high-level feature extraction networks
are  designed  for  each  task  to  extract  task-specific  high-level  features
from  shared  low-level  features,  retaining  commonality  in  low-level
features while aiding the model in better distinguishing feature repre-
sentations  for  different  tasks  at  a  higher  level.  By  simultaneously

Fig. 8. Multi task model feature decomposition visualization flow chart.

identity related features should be included in the hand trace depth
fuzzy features that change with time. Distinguishing between the two
features has a positive effect on identity recognition and time esti-
mation tasks. Therefore, learning two related tasks at the same time
to establish synergies can improve the performance of the model.
2)  Noise is common in infrared images, especially in thermal traces. In
addition  to  the  identity  information  and  time  information,  there
should be other irrelevant information in the hand heat trace. These
information  mainly  include  the  noise  information  present  in  the
image and other small information unrelated to identity recognition
and time estimation, such as the possible small temperature changes
of the same object during the process of leaving thermal traces. These
noises  and  information  can  cause  unnecessary  interference  to  the
identity and time information we extract. Therefore, it is necessary to
add structures to remove noise and other irrelevant information in
order  to  better  obtain  identity  and  time  information  in  hand  heat
trace images. We have learned that soft thresholding is the core of
signal denoising methods. Adding it to the feature extraction module
can  reduce  redundant  information  to  a  certain  extent  (Fu  et  al.,
2019). In order to effectively reduce the redundant information in
the process of hand trace feature extraction, we add soft thresholding
to the feature extraction model and improve the model for hand trace
extraction task.

ExpertSystemsWithApplications255(2024)1245517X. Yu et al.

Table 4
Comparative experiment for loss weights.

λid

1
0.8
1
1.2
1.5
1.2
1.2

λtime

0.01
0.1
0.1
0.1
0.1
0.2
0.05

Acc (%)

75.58
73.14
77.50
77.69
75.77
77.05
77.24

Table 5
Comparison results of different initial loss weights.

λid

0.5
1.2

λtime

0.5
0.1

Acc_id (%)

Acc_time (%)

78.27
81.47

72.95
73.14

learning multiple related tasks, shared bottom-level features can capture
correlations  between  different  tasks,  enabling  the  model  to  achieve  a
synergistic effect and thereby enhance overall performance.

During model training, the SGD and Adam optimization algorithms
are employed to separately train the model for its two tasks, with SGD
showing  relatively  smooth  curves  on  the  loss  graphs  for  both  tasks.
Additionally, common fixed learning rate strategies, warm-up learning
rate strategies, and decay learning rate strategies in multi-task learning
are compared for their effectiveness. Both tasks achieve higher accuracy
curves and lower loss value curves under the decay learning rate strat-
egy. Training multiple tasks simultaneously maintains relatively stable
convergence, ensuring the overall performance of the model.

the two-layer FC network, so that the scaling parameter is scaled to the
range of (0, 1). After that, the threshold is obtained by element level
multiplication of the scaling parameter and the one-dimensional vector
before  the  FC  network.  Finally,  the  output  of  the  soft  thresholding
module is obtained according to the following algorithm.

Algorithm: Soft thresholding

Input: input feature x, threshold T
Step 1: Perform absolute value operation on input features to obtain |x|.
Step 2: According to max{|x| (cid:0) T, 0 }, identify redundant information in the input

feature x.

Step 3: According to the formula of soft thresholding soft(x, T) =

sign(x)max{|x| (cid:0) T, 0 }, transform useful information to very positive or negative
features.

Step 4: Combine with the input feature to obtain the output feature y=soft(x, T) + x.
Output: output feature y

4.3. Feature decomposition

As the hand heat trace change a lot over time, the critical problem of
hand Trace recognition is that the infrared hand trace contour and other
variation  due  to  thermal  diffusion  usually  introduces  the  increasing
intra-class distances. As a result, it is challenging to correctly recognize
two  hand  heat  traces  of  the  same  person  with  a  large  gap,  since  the
mixed hand heat trace representations are severely entangled with un-
related information such as hand contour shape and fuzzy edge changes.
For  this  kind  of  problem,  some  scholars  design  a  linear  factorization
module  to  decompose  the  feature  vectors  into  these  two  unrelated
components  (Gong  et  al.,  2019).  Many  similar  multitask  learning
methods  perform  feature  decomposition  based  on  this  method.
Formally, feature decomposition is to give the feature X extracted from
the input image. The linear factorization module is defined as:

4.2. Residual improvement module based on soft thresholding

X = X1 + X2

(2)

In the proposed network, the mixed features are extracted through
four stages of stacking residual blocks. A residual block is composed of
convolution layer, BN layer, activation layer and one shortcut, as shown
in Fig. 6. Fig. 6a and b shows residual blocks with stride of 2 that result
in different sizes of output feature maps. The residual block reduces the
calculation amount of the following layers by reducing the width of the
output feature map, and promotes the integration of different features
into discriminative features by increasing the number of channels of the
output feature map. Fig. 6c shows a residual block with stride of 1. In the
same  stage,  more  features  can  be  obtained  through  multiple  residual
blocks  with  stride  of  1.  However,  with  this  process,  some  redundant
information will be generated. Soft thresholding has often been used as a
key  step  in  many  signal  denoising  methods  (Donoho,  1995;  Isogawa
et al., 2017). In the deep neural network, it also shows a strong role in
reducing redundancy. We add the soft thresholding method to the re-
sidual block, as shown in Fig. 6c. We only improve the residual block
with stride of 1, and finally form a feature extraction network.

As  for the module of soft thresholding, it can transform useful  in-
formation to very positive or negative features and noise information to
near-zero  features  to  reduce  redundancy. The  function  of  soft  thresh-
olding can be expressed as follows:

⎧
⎨

y =

⎩

x (cid:0) T(x > T)
0(|x| ≤ T )
x + T(x < (cid:0) T)

(1)

where  x  is  the  input  feature,  y  is  the  output  feature,  and  T  is  the
threshold. The soft threshold can be used to set the near-zero features to
zeros. In Fig. 6c, absolute operation and global average pooling (GAP)
are applied to reduce the feature map to a one-dimensional vector. Then,
the one-dimensional vector is propagated into a two-layer FC network.
The number of FC network neurons is equal to the number of channels of
input feature mapping. A sigmoid function is then applied at the end of

X1  and X2  respectively represent the two decomposed components. In
our  study,  the  corresponding  representation  is  the  time  related  and
identity  related  components.  The  feature  decomposition  on  the  two-
dimensional  feature  map  can  use  the  attention  mechanism  to  super-
vise  the  learning  of  multiple  components.  And  decompose  the  mixed
feature-maps at a high-level semantic space.

In this paper, we adopt the sum average of channel attention (CA)
(Hu et al., 2018) and spatial attention (SA) (Lee et al., 2018) to highlight
time related information at both channel and spatial levels (as shown in
Fig.  7).  The  channel  attention  module  uses  spatial  pyramid  pooling
(SPP) to perform parallel operations of maximum pooling and average
pooling on input features, in order to extract richer high-level features
and reduce information loss. The spatial attention module uses global
average  pooling  and  global  maximum  pooling  to  generate  spatial
attention  maps  based  on  the  spatial  relationships  of  features,  high-
lighting the effective information in the feature maps. Formally, we use
a resnet-like backbone to extract mixed feature maps from input image.
The decomposition of mixed feature maps can be defined as follows:

X = X*(fCA(X) + fSA(X)) + X*

(cid:0)

1 (cid:0)

fCA(X) (cid:0)

fSA(X)

)

(3)

where  *  denotes  element-wise  multiplication.  fCA( • ) and  fSA( • )
represent  channel  attention  module  and  spatial  attention  module
respectively. The time related information in the feature maps can be
separated through the attention module supervised by a time estimation
task, and the residual part, regarded as the identity related information,
can be supervised by a hand trace identity recognition task.

Through  the  feature  decomposition  of  a  multitasking  model,  the
features containing overall diverse information are ultimately decom-
posed into identity features for identity recognition and time features for
time  estimation.  In  order  to  more  intuitively  demonstrate  the  feature
separation  effect  of  the  multitasking  model,  visual  processing  is  per-
formed on each major module during the model validation process, and

ExpertSystemsWithApplications255(2024)1245518X. Yu et al.

Table 6
Identity prediction performance in the test set.

Model

Params (M)

Test accuracy (%)

HF-Alex (Chellappa et al., 2017)
HF-Resnet (Chellappa et al., 2017)
HE-CNN
MTLface (Huang et al., 2021)

Alexnet-multi
Resnet34-multi
Resnet18-multi
Efficientnet-b0-multi
Regnet_x_400mf-multi
OUR

6.43
59.75
25.59
50.43

12.76
33.12
23.01
46.50
13.76
52.96

68.20 ± 1.05
70.19 ± 1.50
73.53 ± 2.45
76.07 ± 0.38

72.95 ± 0.41
75.53 ± 0.35
75.83 ± 1.49
69.42 ± 0.73
68.29 ± 0.67
80.17 ± 0.94

Table 7
Departure time estimation performance in the test set.

Model

Error-60 (%)

Error-120 (%)

HF-Alex (Chellappa et al., 2017)
HF-Resnet (Chellappa et al., 2017)
HE-CNN
MTLface (Huang et al., 2021)

Alexnet-multi
Resnet34-multi
Resnet18-multi
Efficientnet-b0-multi
Regnet_x_400mf-multi
OUR

33.37 ± 0.29
37.86 ± 1.56
30.88 ± 1.12
27.82 ± 0.50

29.49 ± 0.51
29.04 ± 1.10
28.03 ± 0.59
34.02 ± 0.74
33.61 ± 0.38
27.25 ± 0.24

9.64 ± 0.76
11.80 ± 0.46
8.70 ± 0.73
9.53 ± 1.09

8.70 ± 0.06
6.77 ± 0.52
7.46 ± 0.94
10.98 ± 0.21
10.36 ± 1.35
8.76 ± 0.16

Fig.  9. Residual  of  departure  time.  The  gray  dotted  line  divides  different
identity categories. The blue dotted line represents the standard deviation. (For
interpretation  of  the  references  to  color  in  this  figure  legend,  the  reader  is
referred to the web version of this article.)

the  partial  feature  maps  obtained  from  the  modules  are  output  and
displayed.  The  following  Fig.  8  shows  the  visualization  feature  flow
diagram of each module in the multitasking model. From the figure, it
can be seen that the image features are deepened layer by layer in the
basic Resnet network section, corresponding to the four feature maps
from left to right in the figure. The deepened features are decomposed
into multiple tasks and provide feature dependencies for different tasks.
It can be seen that the two sets of features after separation have signif-
icant  differences,  and  some  features  also  have  similarities,  indicating
that there are both common feature components and unique specialized
features  between  different  tasks.  Moreover,  the  combined  separated
features can effectively restore the overall features.

Fig. 10. Departure time residual frequency statistics.

4.4. Multi task learning structure of hand trace recognition

To robustly decompose features, we use a time estimation task and
an  identity  recognition  task  to  supervise  the  feature  decomposition.
Specifically,  the  time  related  component  represents  the  trace  feature
that  change  with  time  through  the  time  estimation  task,  while  the
identity  related  component  encodes  the  identity  information.  To  be
clear, we design two classifiers, one is classifier A for time estimation,
and the other is classifier B for identity recognition. A with two linear
layers of 512 and 1 neurons to achieve time regression that learns the
time distribution. The time feature vector is extracted from the linear
layer of 512 neurons, and then the second linear layer is used for time
regression. For the loss function of time estimation, we use the mean
squared error (MSE) to train the time estimation task, as shown in the
formula 4:

Ltime = 1
N

∑N

i=1

(ti (cid:0) ̂ti )2

(4)

where ̂ti  represents the estimated time of the i-th sample, and ti  repre-
sents the ground-truth departure time of the i-th sample.

For  classifier  B,  we  leverage  one  linear  layer  L  of  512  neurons  to
extract  the  identity  feature  vectors.  Considering  the  large  intra-class
distances  of  hand  trace,  the  softmax  loss  of  traditional  CNNs  usually
lacks the ability to distinguish. We use the CosFace loss (Wang et al.,
2018) to supervise the learning of identity feature for identity predic-
tion. The formula of Cosface loss is as follows:

Lid = (cid:0) 1
N

∑

log

i

es(cos(θyi,i)(cid:0) m)

∑

es(cos(θyi,i)(cid:0) m) +

j∕=yiescos(θj,i)

(5)

where  N  is  the  number  of  identities,  yi  is  the  corresponding  identity
label, cos(θj, i) is the cosine of angle between the i-th identity feature and
the j-th weight vector of the classifier. The m a constant margin term
controlling the cosine margin and the s is a constant scaling factor s. The
CosFace loss can introduce much strict constraints to the identity clas-
sification,  and  can  encourage  the  learned  features  to  be  separated
through  a  margin  between  different  identities.  There  is  a  large  intra-
class  distances  in  hand  trace  recognition.  We  expect  to  encourage
powerful  discriminating  information  in  the  learned  features  of  trace
identity recognition by Cosface loss.

The final loss of multi task learning is the weighted sum of two losses.

The final loss is formulated as:

Ltotal = λidLid + λtimeLtime
= λidlcosface(B(Xid), yid) + λtimelmse(A(Xtime), ytime)

(6)

ExpertSystemsWithApplications255(2024)1245519X. Yu et al.

Table 8
Ablation study.  The block_s1  indicates that when the residual
block with stride of 1, a soft thresholding module is added. The
block_s2 indicates that when the block with stride of 2, a soft
thresholding module is added. the block_all indicates that a soft
thresholding module is added to all blocks.

Model

Base
Base + block_s1
Base + block_s2
Base + block_all

Top1-error(%) ± std

21.41 ± 1.13
19.83 ± 0.94
23.74 ± 0.63
23.25 ± 0.77

Fig.  11. Accuracy  curves  of  identity  recognition  for  multi  task  and  single
task models.

Fig. 12. Time estimation accuracy curves for multi task and single task models.

where  the  first  term  is  the  CosFace  loss,  the  second  term  is  the  time
estimation loss, Xid  and Xtime  represent identity related component and
time related component respectively, yid  and Xtime  represent the identity
label and the ground-truth departure time respectively. λ represents the
loss item weight parameter, which controls the balance of different loss
items. We use an approach (Cipolla et al., 2018) to weighs multiple loss
functions  by  considering  the  homoscedastic  uncertainty  of  each  task.
The expression of the loss function is as follows, where σ2
time  are
learnable hyperparameter.
)

id  and σ2

ʹ(cid:0)
Ltotal

σ2
id

, σ2

time

(cid:0)
Ltime + log

)

(cid:0)
+ log

σ2
id

σ2
time

)

(7)

= λidLid + λtimeLtime
Lid + 1
= 1
2σ2
2σ2
id

time

5. Experiment

5.1. Implementation details

Dataset details: Data augmentation methods are applied at training
time  to  improve  hand  heat  trace  recognition  performance.  The  data
augmentation techniques consider the possible natural variables of hand
pose placement and the possible error variables of manual data clipping
processing during the acquisition process, which are mainly reflected in
the  position,  size  and  rotation  angle  of  hand  traces.  Therefore,  the
rotation method enables each sample input to the network to randomly
select a rotation value from the uniform distribution of (cid:0) 10 to 10 for
transformation. A shift method is also applied. The method makes the
trace image of the input network generate random values from 0 to 10
percent shift in the X and Y axis through random clipping. The choice of
random  value  considers  the  integrity  of  traces.  Finally,  we  also  used
horizontal  random  flipping.  Although  there  will  be  a  little  difference
between left-hand and right-hand traces, we ignore the impact of hori-
zontal  flipping  in the  training set. To  sum  up, the  specific  processing
steps are: 1) Align the training data to 224 × 224; 2) Random rotation; 3)
Random horizontal flipping; 4) Randomly crop to 200 × 200; 5) Align
trace image to 112 × 112. In addition, the test set data only aligns the
data to 112 × 112.

The label range of hand trace leaving time is large. During training,
MSE loss is used to calculate the loss of the time estimation task, and
excessive loss value leads to training failure. Therefore, we set a time
scale factor to reduce the ground-truth departure time. We set the time
scaling factor to 30.

Training details: about the setting of relevant super parameters in
the  training  stage.  The  model  is  optimized  by  SGD  with  an  initial
learning rate of 0.0005. Momentum is a training strategy to accelerate
training using the updates in the previous step. The coefficient of the
momentum  is  set  to  0.9  following  the  recommendation  in  (He  et  al.,
2016). L2 regularization is used to reduce the effect of overfitting. L2
regularization adds a penalty term in the objective function to push the
weights to zero. The coefficient of the penalty item of the optimizer is set
to 0.0005.

We trained the model with a batch size of 16 on NVIDIA GTX 2080Ti
GPU.  The  epoch  of  training  stage  is  100.  The  learning  rate  of  model
reduced  by  a  factor  of  0.2,  at  epochs  30,  60,  80  on  train  dataset,
respectively. The multiplicative margin and scale factor of CosFace loss
are set to 0.35 and 64, respectively. All images are aligned to 112 × 112,
and linearly normalized to [(cid:0) 1, 1].

5.2. Learning of loss weight parameters

In multi task learning, the weight of loss value has a great impact on
the  final  performance.  For  the  selection  of  loss  weights,  first,  a
comparative experiment is designed to select the weight parameters that
make  the  model  perform  better  in  recognition  performance.  The
experimental results are shown in the Table 4. When λid is 1.2 and λtime is
0.1, the model can achieve the best accuracy in identity prediction.

id  and  σ2

Then,  according  to  the  weight  adjustment  method  proposed  in
(Cipolla et al., 2018), the model learns the identity prediction classifi-
cation task and the time estimation regression task. We need to deter-
mine  the  initial  values  of  hyperparameter  σ2
time.  Therefore,
different  initial  schemes  are  compared  through  experiments:  1)  Stan-
dard configuration. Both σ2
time are set to 1, and the initial weights
of  the  loss  term  are  0.5  and  0.5,  respectively;  2)  Based  on  the  best
weights in Table 4, set the initial weight value of the loss term toλid = 1.2
andλtime  = 0.1. The comparison results of the two initial weight schemes
in the test set are shown in Table 5. When the initial weight value of the
loss  term  is  set  toλid  = 1.2  andλtime  = 0.1,  the  accuracy  of  heat  trace
identity  prediction  can  be  improved  by  3.2  %  compared  to  another
scheme.

id and σ2

ExpertSystemsWithApplications255(2024)12455110X. Yu et al.

which decompose mixed features into corresponding tasks, can achieve
better results than HF-Alex and HF-Resnet. In terms of classical models,
the performance of the multi task learning models Resnet18-multi and
Resnet34-multi, which use Resnet to extract features, is better than other
models in the identity prediction task. Compared with the best accuracy
of other models, the accuracy of the proposed model in the test set can be
improved by 4.1 %.

5.4. Departure time estimation performance evaluation

In order to evaluate the performance of the model in estimating the
hand heat trace departure time, we choose the same multi task learning
model as the previous part to compare with the proposed model. We
choose  the  residual  of  the  ground-truth  departure  time  and  the  esti-
mated  departure  time  as  the  metric  to  evaluate  the  time  estimation
performance of the model. If the residual of the departure time exceeds
the set value, it is determined that the time estimation is wrong. In this
paper, we choose 60 s and 120 s as the set values to judge the estimation
performance. The model performance is evaluated by the error rate with
the residual within 60 s (Error-60) and the error rate with the residual
within 120 s (Error-120). The results are shown in Table 7. The proposed
method can achieve better performance than other models in Error-60.
The proposed method adopts a structure like to MTLface. By adding soft
thresholding modules and network optimization, the performance of the
model for estimating the hand heat trace departure time is improved.
In  order  to  show  the  performance  of  time  estimation  more  intui-
tively, we analyze the residual between the estimated departure time
and the ground-truth departure time, and give the residual graph of the
test sample, as shown in Fig. 9. The standard deviation of the proposed
model in the test set is 60.34 s. It can be seen from the figure that for
most samples, the residuals between the estimated time and the ground-
truth departure time are within 150 s.

In  order  to  further  explain  the  experimental  results,  we  carry  out
frequency statistics on the residual values of all samples, and draw the
corresponding distribution curve. As shown in Fig. 10, the results from
the bar chart show that there are a large number of data points with
residuals between (cid:0) 120 and 120, and the distribution of the data shows
a clear central trend.

In order to comprehensively showcase the superior performance of
the method proposed in this paper compared to manual experiments, we
introduced visual comparisons to present the time estimation results in
an intuitive and clear manner. In the manual experiments, we selected
twelve  sequences  consisting  of  six  identity  categories  and  four  pose
categories, and invited six experienced subjects for testing. The testing
process involved selecting twelve reference images and corresponding
uniformly spaced four test images.

During  the  tests,  we  first  presented  the  reference  images  to  the
subjects, followed by randomly presenting a test image and requesting
the subjects to observe and determine which reference image it corre-
sponded to. The subjects recorded the recognition time and results. After
all test images were presented, we organized the results and compared
them with the model proposed in this paper, presenting the comparison
results in an intuitive visual interface.

Through visual comparisons, it is clear that the model proposed in
this paper exhibits significantly better performance in terms of recog-
nition  accuracy  and  time  compared  to  manual  experiments.  The
graphical representation provides a more intuitive understanding of the
model’s  performance,  highlighting  its  efficiency  and  reliability  in
fingerprint  recognition  tasks.  This  not  only  confirms  the  model’s  ad-
vantages in improving accuracy and saving human resources but also
provides strong support for its widespread application in practical sce-
narios. Through such visual comparisons, we gain a more comprehen-
sive understanding of the superiority of the new model, providing robust
evidence  for  the  further  development  of  fingerprint  recognition
technology.

Fig.  13. Loss  curves  for  multi  task  and  single  task  time  estimation  on  the
training set.

Fig. 14. Accuracy curves of multi task and single task time estimation under
the training set.

Table 9
Performance of more datasets.

DatasetV2

Train:3231

Test:1682

Model

Acc(%)

Error-60(%)

Error-120(%)

Resnet18-multi
Resnet34-multi
Alresnet34-multi
Our

81.65 ± 1.60
80.99 ± 2.18
77.66 ± 2.39
75.96 ± 0.23

21.88 ± 1.03
22.26 ± 0.87
22.79 ± 0.74
20.95 ± 0.34

5.95 ± 0.69
6.20 ± 0.97
6.14 ± 0.25
4.92 ± 0.37

DatasetV3

Train:3196

Test:1717

Model

Acc(%)

Error-60(%)

Error-120(%)

Resnet18-multi
Resnet34-multi
AIresnet34-multi
Our

79.61 ± 1.63
80.49 ± 0.08
83.25 ± 1.17
83.48 ± 0.98

18.66 ± 1.15
22.25 ± 0.71
18.87 ± 1.95
18.00 ± 0.83

2.85 ± 0.53
5.11 ± 0.92
2.54 ± 0.10
3.55 ± 0.57

5.3. Identity prediction performance evaluation

In order to verify the performance of the proposed model in hand
trace identity prediction, we compared the proposed method with other
multi task learning models. The experiment is divided into two parts. We
choose the models that have shown competitive performance in multi
task learning tasks for comparison. In addition, we modify the classic
classification models in Section 2 to multi task models for comparison.
The experimental results are shown in Table 6. From the table, we can
find that in terms of multi task learning model, HE-CNN and MTLface,

ExpertSystemsWithApplications255(2024)12455111X. Yu et al.

Table 10
Performance in all datasets.

Table 11
Artificial experiment results.

Model

Acc(%)

Error-60(%)

Error-120(%)

Resnet18-multi
Resnet34-multi
Alresnet34-multi

79.03 ± 2.41
79.18 ± 2.46
78.99 ± 3.08

22.86 ± 2.93
24.51 ± 3.20
23.16 ± 3.67

5.42 ± 1.92
6.03 ± 0.69
6.07 ± 2.85

Our

79.87 ± 3.08

22.06 ± 3.86

5.74 ± 2.20

5.5. Ablation study

In order to gain a deeper understanding of the impact of the proposed
multitasking network on identity recognition and time estimation tasks,
we conducted a series of ablation experiments. Compare the effective-
ness of the proposed soft threshold structure at different positions in the
residual block to obtain the optimal soft threshold residual structure. In
addition,  to  verify  the  effectiveness  of  the  multitasking  model,  we
compared  the  multitasking  network  with  the  single  task  network.  By
removing network structures unrelated to identity recognition or time
estimation  from  the  multitasking  network,  we  obtained  the  corre-
sponding single task control model, and then compared and evaluated
the effectiveness of the multitasking network.

5.5.1. Soft threshold ablation experiment

The improved residual block based on soft thresholding is used to
reduce  the  redundant  information  in  the  proposed  model,  which  can
improve the recognition performance of the model. In this part, in order
to  investigate  the  effect  of  the  module  in  the  proposed  model,  we
inserted soft thresholding modules at different locations in the network
for ablation research. The results are shown in Table 8. We believe that
the  convolution  operation  between  the  same  dimensions  at  the  same
stage will generate redundant information, so we add the soft thresh-
olding  module  to  the  residual  block  with  the  stride  of  1.  The  results
proved  this  view  in  the  hand  trace  recognition  task.  When  the  soft
thresholding module is added to the residual block with stride of 1, the
final  error  rate  is  the  lowest,  and  the  recognition  performance  of  the
model is the best.

5.5.2. Multi task ablation experiment

(1)  Single task network for comparing identity recognition tasks

For the identity recognition task, we designed a single-task network
that removed the time estimation-related components from the multi-
task network, retaining only the identity recognition part after the multi-
head attention mechanism. Specifically, we excluded the time estima-
tion network from the multitask model, preserving only the output of the
multi-head attention module. This output was subtracted from the fea-
tures obtained by the backbone and used as the model’s feature output
for identity recognition classification. The identity recognition perfor-
mance of the multitask and single-task models is illustrated in Fig. 11.
The blue and orange lines in the figure represent the validation ac-
curacy  curves  of  the  multi  task  model  and  the  single  task  identity
recognition  model  during  the  training  process.  The  horizontal  axis
represents  the  training  batch,  and  the  vertical  axis  represents  the
recognition accuracy of the model on the validation set. The accuracy is
the  ratio  of  the  number  of  correctly  recognized  samples  to  the  total
number of samples.

On  the  validation dataset,  compared to the  single-task model, our
proposed  multitask  model  demonstrated  higher  identity  recognition
accuracy. This suggests that the introduction of the time estimation task
had a positive impact on network training.

(2)  A Single Task Network for Comparing Time Estimation Tasks

ExpertSystemsWithApplications255(2024)12455112X. Yu et al.

For the time estimation task, we constructed a single-task network by
removing the identity recognition network from the multitask network.
The  overall  network  architecture  involved  feature  extraction  by  the
backbone, further feature extraction through the multi-head attention
mechanism, and application to the regression task for time estimation.
The  time  estimation  results  of  single-task  and  multitask  models  are
depicted in Fig. 12.

The blue and gray lines in the figure represent the validation accu-
racy  change  curves  of  the  multi  task  model  and  the  single  task  time
regression model during the training process, with an allowable error
time of 60 s. The orange and yellow lines represent the accuracy change
curves  of the  two  models with  an allowable  error time  of 120  s.  The
horizontal axis represents the training batch, and the vertical axis rep-
resents the recognition accuracy of the model within the allowable error
time N on the validation set. The allowable error time N indicates that
the  prediction  is  considered  correct  when  the  error  between  the  pre-
diction and the true value is within N.

From the graph, it can be observed that the multitask model’s per-
formance in time estimation is slightly lower than that of the single-task
model, but the difference is not significant. This may be due to the fact
that in the multitask network structure, only the addition of time fea-
tures onto identity features occurs, with identity features having mini-
mal impact on time features, resulting in similar outcomes for multitask
and single-task modes.

From the accuracy curves of single task identity recognition, single
task time estimation, and multi task models on the validation set, it can
be seen that the curve complexity of the multi task model is significantly
higher than that of the single task model, with significant fluctuations.
We  speculate  that  the  curve  fluctuation  of  the  multitasking  model  is
significant, possibly due to the complex interrelationships and compe-
tition between tasks, which may cause interference from different tasks
in  the  multitasking  learning  model.  During  the  optimization  process,
adjusting one task may affect the performance of other tasks, leading to a
more complex and unstable optimization process.

In addition, in the loss curve of the training set in Fig. 13 and the time
estimation accuracy curve in Fig. 14, we observed that the curve of the
multitasking model had two stages of jumps, ultimately exceeding the
accuracy of the single task model. The reason for this phenomenon may
also  come  from  the  mutual  influence  between  tasks  in  multi  task
learning, as well as the different data distributions involved in tasks in
multi task learning, which in turn affects the convergence process of the
model.  At  the  same  time,  it  may  also  come  from  the  adjustment  of
learning  rate  during  the  training  process.  The  changing  learning  rate
may have a greater impact on the multi task model. As can be seen from
Fig. 14, there is also a small step in the accuracy of the single task model
on the training set.

In summary, the multitask model exhibits significant fluctuations in
accuracy during the early stages of training, possibly due to the need to
balance and compete between different tasks. The model may initially
overfit  certain  tasks  and  underfit  others,  resulting  in  significant  loss
curve fluctuations. However, by observing the time estimation accuracy
curve on the training set, we can infer that the multitask model gradu-
ally overcomes these fluctuations, ultimately achieving higher accuracy.
This  indicates  that  multitask  learning  has  potential  advantages  in
identity recognition and time estimation tasks, enabling the network to
learn task-related features more comprehensively and robustly.

5.5.3. Summary of ablation research

Through  ablation  experiments,  the  optimization  of  the  residual
structure of the model and the study of the impact of multi task networks
on identity recognition and time estimation tasks were achieved. In the
soft threshold ablation experiment, a detailed ablation study was con-
ducted  on  the  model  structure by  inserting  soft  threshold  modules  at
different positions in the residual network. The performance differences
between  the  original  ResNet  structure  and  the  improved  version  of
ResNet  with  the  addition  of  soft  threshold  nonlinear  transformation

Fig. 15. Data set grouping. Each row represents six identity categories, and the
four  columns  divided  by  black  lines  represent  the  types  of  hand  traces.  The
hand traces displayed represent their respective image sequences.

layer in identity recognition and time estimation tasks were compared,
and  it  was  evaluated  that  adding  soft  threshold  residual  modules  at
appropriate positions can effectively reduce redundant information. In
the multi task ablation experiment, by gradually removing components
unrelated  to  specific  tasks  from  the  multi  task  network,  a  single  task
control model for identity recognition task and comparison time esti-
mation task was constructed, and the performance differences between
the multi task network and the single task network were compared and
analyzed,  verifying  the  effectiveness  of  the  multi  task  model  and  the
positive impact of time estimation on network training.

5.6. Comparison of more datasets

We utilize the deep learning method to estimate the departure time
of hand traces. The experimental results show that the accuracy of the
residual within 60 s can only reach 73 %, which is far from the actual
application needs. Considering the requirement of deep learning for the
number of samples, we think that the division of training set and test set
may limit the performance of the model. Table 9 shows our experiment
to divide the dataset again. We randomly divide the dataset again ac-
cording to the division method in the design of dataset in Section 3, and
build  DatasetV2  and  DatasetV3.  We  choose  resnet18-multi,  resnet34-
multi  and  Alresnet34-multi,  which  have  remarkable  performance  in
time  estimation  and  identity  prediction,  as  the  comparison  model.
Alresnet34-multi  is  the  model  modified  by  MTLface  according  to  the
hand trace recognition task. The data in the table shows that there are
significant differences in model performance under different datasets. In
terms  of  the  estimation  performance  of  departure  time,  the  proposed
method can achieve better performance than other models. In terms of
identity  prediction  performance,  the  prediction  accuracy  of  the  pro-
posed method in the test set of DatasetV2 is only 75.96 %. Finally, we
calculated the average identity prediction accuracy and the error rate of
average departure time estimation in the above three datasets, as shown
in  Table  10.  Our  method  can  achieve  better  performance  than  other
models.

5.7. Artificial experiments

To  study  the  recognition  performance  of  the  model,  artificial  ex-
periments  were  added  to  the  dataset  as  a  comparative  experiment.
Similar  and  close-to-real  fingerprint  images  were  selected  from  the
dataset,  and  six  subjects  with  relevant  recognition  experience  were
chosen for manual recognition. The artificial experiment data included
twelve  sequences,  comprising  six  identity  categories  and  four  pose

ExpertSystemsWithApplications255(2024)12455113X. Yu et al.

Fig. 17. Average error rate of different methods in the dataset.

six subjects. From the statistically organized data, it can be observed that
in the manual experiments, the average recognition time was relatively
long, and the accuracy was not satisfactory. The table shows that the
recognition performance in manual experiments exhibited some insta-
bility,  with  both  long  recognition  times  and  lower  accuracy  rates.  In
comparison, the proposed model can process multiple images in a single
recognition operation and performs significantly better in terms of ac-
curacy and speed. This further validates the superiority of the proposed
model  in  the  field  of  fingerprint  recognition,  not  only  enhancing
recognition accuracy but also significantly saving human resources.

6. Analysis and discussion

In  this  section,  we  further  analyze  the  performance  problems  of
identification and departure time estimation based on the experimental
results  in  the  previous  section.  We  start  from  three  parts:  analysis  of
dataset grouping, analysis of identity prediction results, and analysis of
departure time estimation.

6.1. Dataset grouping

In the last part of the experiment, we randomly grouped the datasets
and built DatasetV2 and DatasetV3 to compare the performance of the
model.  In  Fig.  15,  we  gave  the  specific  division  of  the  datasets.  We
randomly select 2 groups as the training set and 1 group as the test set
from 3 groups of sequences with the same identity and the same hand
type.  The  changes  of  the  two  datasets  are  shown  in  red  boxes  in  the
figure. Among them, category 3, 4 and 6 have the least obvious distri-
bution change in the two datasets, while category 5 has the most obvious
change.

6.2. Analysis of identity prediction results

Based  on  the  error  rate  of  identity  prediction  obtained  from  the
experiment, we analyzed it from three aspects: identity category, hand
type and different time stages.

Identity category: different division of datasets enable us to obtain
completely different model performance. The performance of DatasetV3
is like to that of DatasetV1, and our method is relatively competitive.
However, the performance on DatasetV2 is different than expected. In
Fig. 16, we visualize the distribution of identity prediction error samples
on DatasetV2 and DatasetV3. It is found from the figure that the training
effect of the fifth and sixth categories is not ideal, and the predictions of
the fifth and sixth categories are almost wrong.

Further, we calculate the average error rate of different methods in
DatasetV2 and DatasetV3, as shown in Fig. 17. The recognition perfor-
mance of all methods in Category 5 and Category 6 is the worst, and they
are generally unable to effectively learn the feature of categories. We

Fig.  16. Distribution  map  of  identity  prediction  error  samples.  Four  colors
correspond  to  four  methods.  Each  method  is  trained  and  tested  3  times  ac-
cording  to  the  same  settings.  The  dotted  line  represents  the  boundary  of
6 categories.

categories. During the process of the artificial experiment, twelve real
fingerprint images were initially selected from the dataset as reference
images.  Four  corresponding  test  images  were  then  selected  for  each
reference image, ensuring a large and uniform span. The reference im-
ages were presented to the subjects, followed by the random presenta-
tion  of  a  test  image.  The  six  subjects  observed  the  test  image  and
determined  its  corresponding  reference  image,  recording  the  recogni-
tion time and results. After the conclusion of the artificial experiment,
the results were statistically organized, and the results for the twelve
sequence groups are presented in the Table 11 below.

Subjects observed the test images, selected the images closest to re-
ality, recorded the time for each recognition, and assessed the accuracy
of the results. To comprehensively compare the recognition performance
of the proposed model with manual experiments, further analysis of the
experimental results was conducted. In the manual experiments, there
were certain differences in the recognition time and accuracy among the

ExpertSystemsWithApplications255(2024)12455114X. Yu et al.

think  that  the  dataset  processing  may  not  be  comprehensive  enough,
because the third hand type of category 6 can find that the hand posture
has obvious deviation, and there may also be the problem of uneven data
distribution.

Hand type: in order to analyze the problem of identity recognition
errors  more  comprehensively,  we  give  the  recognition  error  rates  of
different hand types in the three test sets, as shown in Fig. 18. The fre-
quency of training samples about different hand types in each dataset is
balanced. However, the error rate of RFT has the most significant change
in  the  three  test  sets,  indicating  that  the  division  of  datasets  has  an
impact on the recognition performance. The error rate of RFA is rela-
tively high in the three test sets, indicating that there are problems with
this type of data, and the model cannot learn this type of hand trace
feature well.

Influence  of  time:  we  believe  that  for  samples  with  clear  hand
traces, the identity prediction task can classify more accurately, while
for fuzzy samples, the prediction accuracy will decline. The error rate of
samples with different departure times in the identity prediction task has
obvious differences. Therefore, according to the sample distribution of
different training sets and test sets, we divide the departure time into six
groups  (0–2,  2–4,  4–6,  6–8,  8–10,  10-)  to  calculate  the  error  rate  of
identity prediction in different time groups, as shown in Fig. 19. In the
figure,  we  show  the  frequency  of  training  sample  categories  and  the
identity prediction performance of different models in the dataset. The
prediction accuracy of the model decreases with time. And the proposed
method  can  achieve  competitive  prediction  accuracy  in  the  period  of
2–10  min.  In  all  time  group,  the  identity  prediction  accuracy  of  the
proposed model is also better than that of comparison models.

6.3. Analysis of departure time estimation results

Generally, the hand heat trace can last for 10–15 min. In combina-
tion with the actual criminal investigation application requirements, the
estimated departure time of the trace at the scene of the crime does not
need to be accurate to the second. It is usually required to estimate a
certain  time  period.  Therefore,  we  divide  the  residual  graph  of  the
experimental part into four time stages (0–3 min, 3–6 min, 6–9 min and
9 min later) for further analysis, as shown in Fig. 20. The standard de-
viation of departure time of the four stages are 26.55 s, 54.92 s, 71.67 s
and  71.93  s  respectively.  This  indicates  that  the  time  estimation  per-
formance of the model will gradually decrease with the increase of time.
In addition, we found that the residual value of each test image sequence
has an obvious trend of change. In the same sequence, the residual value
will gradually increase with time. In different time stages, the residual
values  in  the  same  sequence  are  almost  all  positive  or  negative,  not
distributed on both sides of the residual value of 0.

Fig. 18. Recognition Performance of Different Hand Types in three datasets.

Fig. 19. Recognition performance of different time groups in three datasets.

ExpertSystemsWithApplications255(2024)12455115X. Yu et al.

Fig. 20. Residual for departure time at different time stages. (a) 0 to 3 min; (b) 3 to 6 min; (c) 6 to 9 min; (d) After 9 min.

7. Conclusion

The heat trace recognition technology is a potential criminal inves-
tigation method, which can predict the identity of the target and esti-
mate the departure time according to the captured heat trace. However,
due to the influence of thermal diffusion, the thermal traces captured are
generally  deep  fuzzy.  How  to  extract  time  invariant  identity  features
from the deep fuzzy thermal traces is of great significance for thermal
trace recognition. To solve the problem of hand heat trace recognition,
we  propose  a  time  invariant  hand  heat  trace  multi  task  recognition
model (MTLHand) based on the deep learning method, which can pre-
dict the hand identity and estimate the hand departure time. The soft
thresholding module is integrated into the deep structure of the network
to eliminate redundant information. Based on the attention mechanism,
the extracted mix deep features are divided into time related features
and identity related features to improve the recognition performance of
the model.

It is generally unable to achieve competitive performance for a single
identity recognition task through the classic deep network. Compared
with these models, it is verified that the proposed MTLHand can further
improve the performance of identity recognition by using time related
information. The performance of the proposed method is 11.11 % higher
than that of the classic deep learning model. In addition, under the same
learning  models,
conditions,  compared  with  other  multi  task
MTLHand’s average identity prediction performance and average time
estimation  performance  are  improved  by  0.88  %  and  2.45  %  respec-
tively.  Therefore,  using  multi  task  learning  is  an  effective  means  to
further improve the performance of hand trace identification.

In this study, we have established a more comprehensive hand heat
trace dataset, which includes six types of identities and the departure
time  of  heat  traces,  which  can  effectively  promote  the  study  of  hand
traces.  However,  from  the  training  and  testing  of  the  model,  the
collection method of the dataset needs to be improved. Some informa-
tion cannot be effectively learned, which is related to the collection and

processing of the dataset. A set of data screening standards needs to be
established to solve this problem.

Our aim is to build a model that can be applied to the identification of
heat trace and the estimation of departure time in the actual criminal
investigation scene. But the current recognition accuracy cannot meet
the requirements of practical applications. We expect that in the future
research, by optimizing the structure of the model, we can design a multi
task  learning  model  that  is  more  suitable  for  hand  trace  recognition,
learn  time  related  information  and  identity  related  information  more
effectively, and improve the accuracy of time estimation and identity
recognition.

CRediT authorship contribution statement

Xiao  Yu:  Conceptualization,  Software,  Data  curation,  Writing  –
original draft, Writing – review & editing, Validation, Formal analysis,
Methodology,  Supervision.  Xiaojie  Liang:  Data  curation,  Writing  –
original draft, Writing –  review &  editing, Methodology, Supervision,
Project administration, Software. Zijie Zhou: Writing –  original draft,
Writing – review & editing, Visualization, Project administration, Soft-
ware. Baofeng Zhang: Funding acquisition, Investigation, Resources.

Declaration of competing interest

The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.

Data availability

Data will be made available on request.

ExpertSystemsWithApplications255(2024)12455116X. Yu et al.

Acknowledgments

This  research  is  supported  by  the  Tianjin  Postgraduate  Scientific
Research Innovation Project (No. 2020YJSB077). The Tianjin University
of  Technology  Textbook  Construction  Foundation  under  Project  (No.
JC20-11). The Tianjin University of Technology First-class Curriculum
Cultivation Fund (Artificial Intelligence).

References

Ai, J., Hu, M. H., Zhai, G. T., Zhang, X. P., Wang, Y. L., Cai, L. M., Li, Q. L., & Sun, W. Q.
(2020). Rapidly developing human heat residue model under various conditions
based on Fluent and thermal video. Infrared Physics & Technology, 110, Article
103468.

Ali, M. M., Hashim, N., Lasekan, O., & Abd Aziz, S. (2020). Emerging non-destructive
thermal imaging technique coupled with chemometrics on quality and safety
inspection in food and agriculture. Trends in Food Science & Technology, 105,
176–185. ISSN 0924-2244.

Abdelrahman, Y., Khamis, M., Schneegass, S., & Alt, F. (2017). Stay Cool! Understanding
thermal attacks on mobile-based user authentication. In Proceedings of the 2017 CHI
conference on human factors in computing systems, Denver, CO, USA, 6–11 May ;
pp. 3751–3763.

Buddharaju, P., Pavlidis, I. T., Tsiamyrtzis, P., & Bazakos, M. (2007). Physiology-based
face recognition in the thermal infrared spectrum. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 29(4), 613–626.

Chen, J. Z., Jin, D. R., Liu, Z. C., Wang, Y. Y., Yang, F., & Bai, X. Z. (2022). Light transport
induced domain adaptation for semantic segmentation in thermal infrared urban
scenes. IEEE Transactions on Intelligent Transportation Systems, 23(12), 23194–23211.
Chellappa, R., Patel, V. M., & Ranjan, R. (2017). Hyperface: A deep multi-task learning

framework for face detection, landmark localization, pose estimation, and gender
recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(1),
121–135.

Cipolla, R., Gal, Y., & Kendall, A. (2018). Multi-task learning using uncertainty to weigh
losses for scene geometry and semantics. Proceedings of the IEEE conference on
computer vision and pattern recognition. 7482-7491.

Chantaf, S., Hilal, A., & Elsaleh, R. (2020). Palm vein biometric authentication using

convolutional neural networks. In Proceedings of the 8th International Conference
on Sciences of Electronics, Technologies of Information and Telecommunications
(SETIT’18), Vol. 1 (pp. 352-363). Springer International Publishing.

Donoho, D. L. (1995). De-noising by soft-thresholding. IEEE Transactions on Information

Theory, 41(3), 613–627.

Fitch, A., & Touroo, R. (2018). Crime scene findings and the identification, collection,

and preservation of evidence. Veterinary Forensic Pathology, 1, 9–25.

Fu, X. Y., Zhao, M. H., Zhong, S. S., Tang, B. P., & Pecht, M. (2019). Deep residual

shrinkage networks for fault diagnosis. IEEE Transactions on Industrial Informatics, 16
(7), 4681–4690.

Gong, D. H., Li, Z. F., Wang, H., & Liu, W.(2019). Decorrelated adversarial learning for

age-invariant face recognition. Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 3527-3536.

Girshick, R., Kosaraju, R. P., Radosavovic, I., Doll´ar, P., & He, K.M.(2020). Designing

network design spaces. Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 10428-10436.

Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. Proceedings of the

IEEE conference on computer vision and pattern recognition. 7132-7141.

Hou, F. J., Zhang, Y., Zhou, Y., Zhang, M., Lv, B., & Wu, J. Q. (2022). Review on infrared

imaging technology. Sustainability, 14(18), 11161.

Hinton, G. E., Krizhevsky, A., & Sutskever, I. (2017). Imagenet classification with deep

convolutional neural networks. Communications of the ACM, 60(6), 84–90.
Hu, H., Li, H., & Zou, H. (2017). Modified hidden factor analysis for cross-age face

recognition. IEEE Signal Processing Letters, 24(4), 465–469.

Hu, H., Li, H., & Yip, C. (2018). Age-related factor guided joint task modeling

convolutional neural network for cross-age face recognition. IEEE Transactions on
Information Forensics and Security, 13(9), 2383–2392.

Hu, J., Pu, Z. Y., Zhuang, Y. F., & Wang, Y. H. (2021). Illumination and temperature-

aware multispectral networks for edge-computing-enabled pedestrian detection.
IEEE Transactions on Network Science and Engineering, 9(3), 1282–1295.

He, K. M., Ren, S. Q., Zhang, X. Y., & Sun, J. (2016). Deep residual learning for image
recognition. Proceedings of the IEEE conference on computer vision and pattern
recognition. 770-778.

Hu, M. H., Li, D., Zhai, G. T., Fan, Y. Z., Duan, H. Y., Zhu, W. H., Yang, X. K., & Yang, Y.
(2018). Combination of near-infrared and thermal imaging techniques for theremote
and simultaneous measurements of breathing and heart rates under sleep situation.
PLoS ONE, 13(1), Article e0190466.

Hu, M. H., Li, D., Zhang, X. P., Zhai, G. T., & Yang, X. K. (2019). Physical password

breaking via thermal sequence analysis. IEEE Transactions on Information Forensics
and Security, 14(5), 1142–1154.

Hu, M. H., Li, D., Xu, Z. Y., Wang, Q. C., Yao, N., & Zhai, G. T. (2020). Estimating

departure time using thermal camera and heat traces tracking technique. Sensors, 20
(3), Article 782.

Huang, Z., Shan, H., & Zhang, J. (2021). When age-invariant face recognition meets face
age synthesis: A multi-task learning framework. Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 7282-7291.

Isogawa, K., Ida, T., Shiodera, T., & Takeguchi, T. (2017). Deep shrinkage convolutional
neural network for adaptive noise reduction. IEEE Signal Processing Letters, 25(2),
224–228.

Lee, H. J., & Wu, C. (2022). Learning age semantic factor to enhance group-based

representations for cross-age face recognition. Neural Computing and Applications,
1–12.

Lee, J. Y., Park, J., Woo, S. H., & Kweon, I. S. (2018). Cbam: Convolutional block
attention module. Proceedings of the European conference on computer vision
(ECCV). 3-19.

Lavergne, L., Boivin, R., Baechler, S., Jeuniaux, P., Fiola, K., S´eguin, D., Lefebvre, J. F., &
Milot, E. (2022). Determining the impact of unknown individuals in criminality
using network analysis of DNA matches. Forensic Science International, 331, Article
111142.

Le, Q., & Tan, M. (2019). Efficientnet: Rethinking model scaling for convolutional neural

networks. In International conference on machine learning (pp. 6105–6114). PMLR.

Li, X., & Chidurala, V. (2021). Occupancy estimation using thermal imaging sensors and

machine learning algorithms. IEEE Sensors Journal, 21(6), 8627–8638.

Li, X., & Chidurala, V. (2022). Detection of moving objects using thermal imaging sensors

for occupancy estimation. Internet of Things, 17, Article 100487.

Majumdar, G., Bhowmik, M. K., Gogoi, U. R., & Ghosh, A. K. (2019). Evaluating the

efficiency of infrared breast thermography for early breast cancer risk prediction in
asymptomatic population. Infrared Physics & Technology, 99, 201–211.

Mowery, K., Meiklejohn, S., & Savage, S. (2011). Heat of the moment: Characterizing the

efficacy of thermal camera-based attacks. In Proceedings of the 5th USENIX
conference on Offensive technologies, San Francisco, CA, USA, 8–11 August; p. 6.
Szkuta, B., Meakin, G. E., Van Oorschot, R. A. H., Kokshoorn, B., & Goray, M. (2019).

DNA transfer in forensic science: A review. Forensic Science International: Genetics, 38,
140–166.

Scebba, G., Poian, G., & Karlen, W. (2021). Multispectral video fusion for non-contact

monitoring of respiratory rate and apnea. IEEE Transactions on Biomedical
Engineering, 68(1), 350–359.

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale

image recognition. arXiv preprint arXiv,1409.1556.

Wang, H., Wang, Y. T., Zhou, Z., Ji, X., Gong, D. H., Zhou, J. C., Li, Z. F., & Liu, W.

(2018). Cosface: Large margin cosine loss for deep face recognition. Proceedings of
the IEEE conference on computer vision and pattern recognition. 5265-5274.
Zhou, Z., Zhang, B., & Yu, X. (2021). Infrared handprint classification using deep

convolution neural network. Neural Processing Letters, 53, 1065–1079.

Zhou, Z., Zhang, B., & Yu, X. (2022). Immune coordination deep network for hand heat

trace extraction. Infrared Physics & Technology, 127, Article 104400.

ExpertSystemsWithApplications255(2024)12455117