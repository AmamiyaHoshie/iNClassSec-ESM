# iNClassSec-ESM: Non-Classical secreted protein predictor based on multimodal protein language model
## Introduction
Non-classical secreted proteins (NCSPs) are a class of proteins lacking signal peptides, secreted
by Gram-positive bacteria through non-classical secretion pathways. With the increasing demand
for highly secreted proteins in recent years, non-classical secretion pathways have received more
attention due to their advantages over classical secretion pathways (Sec/Tat). However, because the
mechanisms of non-classical secretion pathways are not yet clear, identifying NCSPs through biolog-
ical experiments is expensive and time-consuming, making it imperative to develop computational
methods to address this issue. Existing NCSP prediction methods mainly use traditional handcrafted
feature extraction to represent proteins from sequence information, which limits the models’ ability to
capture complex protein characteristics. In this study, we propose a novel NCSP predictor, iNClassSec-
ESM, which combines deep learning with traditional classifiers to enhance prediction performance.
iNClassSec-ESM integrates an XGBoost model trained on comprehensive handcrafted features and a
Deep Neural Network (DNN) trained on hidden layer embeddings from the protein language model
(PLM) ESM3. The ESM3 is the recently proposed multimodal PLM and has not yet been fully
explored in terms of protein representation. In this study, we extracted hidden layer embeddings from
ESM3 as inputs for multiple classifiers and deep learning networks, and compared them with existing
PLMs. 
<div align=center><img  src ="https://github.com/AmamiyaHoshie/img-repo/blob/main/iNCS-ESM.png" alt="Framework of iNClassSec-ESM"></div>
Benchmark experiments indicate that iNClassSec-ESM outperforms most of existing methods
across multiple performance metrics and could serve as an effective tool for discovering potential
NCSPs. Additionally, the ESM3 hidden layer embeddings, as an innovative protein representation
method, show great potential for the application in broader protein-related classification tasks.
