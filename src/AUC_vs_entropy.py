import numpy
import random
import math
from sklearn import linear_model
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import pickle
import entropy
import pca2
import timeit
dataset="none"
"""
dataset="pima"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'pima.csv',delimiter=',')
rows=readdata.shape[0]
train_data=readdata[0:10,:]
test_data=readdata[10:15,:]
"""

dataset="leukemia"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'lukemia.csv',delimiter=',')
train_data = numpy.transpose(readdata)
readdata = numpy.loadtxt(path+'lukemia_test.csv',delimiter=',')
test_data = numpy.transpose(readdata)

"""
dataset="lung"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'lungCancer_train.csv',delimiter=',')
train_data = numpy.transpose(readdata)
readdata = numpy.loadtxt(path+'lungCancer_test.csv',delimiter=',')
test_data = numpy.transpose(readdata)
"""
"""
dataset="arcene"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'arcene_train.csv',delimiter=',')
train_data = readdata
readdata = numpy.loadtxt(path+'arcene_test.csv',delimiter=',')
test_data = readdata
"""
"""
dataset="madelon"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'madelon_train.csv',delimiter=',')
train_data = readdata
readdata = numpy.loadtxt(path+'madelon_test.csv',delimiter=',')
test_data = readdata
"""
"""
dataset="spam"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'spambase_shuffle.csv',delimiter=',')
rows=readdata.shape[0]
train_data=readdata[0:3500,:]
test_data=readdata[3500:,:]
"""
"""
dataset="colon"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'colon_shuffle.csv',delimiter=',')
rows=readdata.shape[0]
train_data=readdata[0:31,:]
test_data=readdata[31:,:]
"""
"""
dataset="prostate"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'prostate_shuffle.csv',delimiter=',')
rows=readdata.shape[0]
train_data=readdata[0:51,:]
test_data=readdata[51:,:]
"""
"""
dataset="sonar"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path'sonardata_shuffle.csv',delimiter=',')
train_data=readdata[0:150,:]
test_data=readdata[150:,:]
"""
"""
dataset="mfeat"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'zero_one.csv',delimiter=',')
train_data=readdata[range(0,150) + range(200,350),:]
test_data=readdata[range(150,200) + range(350,400),:]
"""
"""
dataset="ionosphere"
path='/home/debajyoti/Data/'+dataset+'/'
readdata = numpy.loadtxt(path+'ionosphere.csv',delimiter=',')
rows=readdata.shape[0]
train_data=readdata[0:200,:]
test_data=readdata[200:,:]
"""
n=train_data.shape[1]
m=test_data.shape[0]

print n, m 

y_train=train_data[:,0]
y_test=test_data[:,0]

print y_train
print y_test

indexset=[]
models=[]
wil_row=[]



pca_time=0.0
entropy_time=0.0
entropy2_time=0.0
#n=10

if n>1000:
 basen=int(0.01*n)
else:
 basen=int(math.ceil(0.2*n))
print n
print 'basen=',basen
filename_init_attr=path+dataset+'.init'
#list_rand=random.sample(range(1,n), basen)
list_rand=numpy.loadtxt(filename_init_attr,delimiter=',')
list_rand = numpy.array(list_rand, int).tolist()     
#numpy.savetxt(filename_init_attr,list_rand,delimiter=',')

#print list_inc
#AUC of all features
#x_train=train_data[:,list_inc]
#x_test=test_data[:,list_inc]
#print x_train.shape
# find logistic function
"""
clf = linear_model.LogisticRegression(C=1.0)
clf.fit(x_train,y_train)
dlabel=clf.predict(x_test)
rauc=roc_auc_score(y_test,dlabel)
print rauc
"""
list_inc=[]
list_exc=[]
change=[]
ig=[]
correlation=[]
#clf =LinearSVC()
clf = linear_model.LogisticRegression(C=1.0)
c=[0,0,0,0]

x_train=train_data[:,list_rand]
x_test=test_data[:,list_rand]
clf.fit(x_train,y_train)
#dlabel=clf.predict(x_test) 
 
dec_val_test=[]
"""
for i in range(test_data.shape[0]):
 dec_val_test.append(clf.decision_function(x_test[i,:])[0])
"""
#ptest=clf.predict_proba(x_test)
#dec_val_test=ptest[:,1]
dec_val_test=clf.decision_function(x_test)
      
auc_base=roc_auc_score(y_test,dec_val_test)
print "base:",auc_base
ent=[]
ent_auc=[]
auc_bool=[]
for gene in range(1,n):#build set for every gene
   print gene
   list_inc = list_rand*1 #convert to integer as they are indices, the set includes gene
   
   
   
   if gene not in list_inc:
      list_inc.append(gene)
      
   list_exc=list_inc*1 
   list_exc.remove(gene)#indices that excludes gene
   
   corr=[]
   x=test_data[:,gene]
   for j in list_exc:
      y=test_data[:,j]
      coeff=pearsonr(x, y)
      #print "%.2f" % coeff[0] ,
      corr.append(abs(coeff[0]))
   max_corr=max(corr)
   
   x_train=train_data[:,list_inc]
   x_test=test_data[:,list_inc]
   clf.fit(x_train,y_train)
   #dlabel=clf.predict(x_test) 
   
   #dec_val_train=[]
   #for i in range(train_data.shape[0]):
   # dec_val_train.append(clf.decision_function(x_train[i:])[0])
    
   dec_val_test=[]
   """
   for i in range(test_data.shape[0]):
    dec_val_test.append(clf.decision_function(x_test[i,:])[0])
   """
   #ptest=clf.predict_proba(x_test)
   #dec_val_test=ptest[:,1]
   dec_val_test=clf.decision_function(x_test)
      
   auc_inc=roc_auc_score(y_test,dec_val_test)
   
   
   x_train=train_data[:,list_exc]
   x_test=test_data[:,list_exc]
   clf.fit(x_train,y_train)
   #dlabel=clf.predict(x_test) 
    
   dec_val_test=[]
   """
   for i in range(test_data.shape[0]):
    dec_val_test.append(clf.decision_function(x_test[i,:])[0])
   """
   #ptest=clf.predict_proba(x_test)
   #dec_val_test=ptest[:,1]
   dec_val_test=clf.decision_function(x_test)
   
   auc_exc=roc_auc_score(y_test,dec_val_test)
   min_en1=entropy.entropy_of_vector(dec_val_test,y_test)
   
   #dec_val_train=[]
   #for i in range(train_data.shape[0]):
   # dec_val_train.append(clf.decision_function(x_train[i:])[0])
  
   pca_data = numpy.transpose(numpy.vstack((test_data[:,gene],numpy.array(dec_val_test))))
   pca_vector=pca2.PCA2(pca_data,1)
   min_en2=entropy.entropy_of_vector(pca_vector,y_test)
   
   ch=(auc_inc-auc_exc)*1.0
   chen=(min_en2-min_en1)*1.0
      
   ent.append([gene,chen])
   
   if ch>0.0 and chen>0.0:c[0]+=1
   elif ch<0.0 and chen>0.0:c[1]+=1
   elif ch<0.0 and chen<0.0:c[2]+=1
   elif ch>0.0 and chen<0.0:c[3]+=1
   
   if ch !=0.0 and chen!=0.0:
   #   print gene, ch
      tempch=(ch/(1.0 - auc_exc))*100.0
      if(tempch > 100.0):
         print "HELLO!!!!", auc_inc,auc_exc,ch,auc_base
      change.append(tempch)
      ig.append(chen)
      ent_auc.append([gene,chen,tempch])
      correlation.append(1.0-max_corr)
      if tempch<=0.0:
         auc_bool.append(1)
      else:
         auc_bool.append(0)

print c[1],c[0]
print c[2],c[3]
filename_con=dataset+'.contable'
filename_rank=dataset+'.entrank'
filename_ent_auc=dataset+'.entauc'
numpy.savetxt(filename_con,c,delimiter=',')
title='base AUC=%0.2f' % auc_base
numpy.savetxt(filename_rank, ent, delimiter=',')
numpy.savetxt(filename_ent_auc, ent_auc, delimiter=',')


   
fpr, tpr, thresholds = roc_curve(auc_bool,ig) 
fpr2, tpr2, thresholds2 = roc_curve(auc_bool,correlation)

#k=20
ent=numpy.array(ent)
feat=ent[numpy.argsort(ent[:,1])]
numpy.savetxt('ent', feat, delimiter=',')
#print wil_feat

##add noise
change_noise=[]
ig_noise=[]
correlation_noise=[]
c=[0,0,0,0]

base_size=n
if n>1000:
   base_size=int(0.1*n)
#list_rand=random.sample(range(1,n), 10)*1
list_rand=list_rand*1
x_test=test_data[:,list_rand]

ent=[]
num=base_size

#auc of base
x_train=train_data[:,list_rand]
x_test=test_data[:,list_rand]
clf.fit(x_train,y_train)
dec_val_test=clf.decision_function(x_test)
auc_base=roc_auc_score(y_test,dec_val_test)
min_en1=entropy.entropy_of_vector(dec_val_test,y_test)

for k in range(num):
   print k,"/",num
   sample=random.randint(1,n-1)
   min_sam=int(min(test_data[:,sample]))
   max_sam=int(max(test_data[:,sample]))
   #print min_sam,max_sam
   noiserow_test = numpy.random.uniform(min_sam,max_sam, [test_data.shape[0]])	#sample values range, random sample size
   min_sam=int(min(train_data[:,sample]))
   max_sam=int(max(train_data[:,sample]))
   noiserow_train = numpy.random.uniform(min_sam,max_sam, [train_data.shape[0]])	#sample values range, random sample size
   
   temp_x_test=numpy.column_stack((x_test,noiserow_test))
   temp_x_train=numpy.column_stack((x_train,noiserow_train))
   
   #correlation
   corr=[]
   x=numpy.asarray(noiserow_test)
   for j in list_rand:
      y=test_data[:,j]
      coeff=pearsonr(x, y)
      if coeff=="nan": coeff=0.0
      #print "%.2f" % coeff[0] ,
      corr.append(abs(coeff[0]))
   max_corr=max(corr)
   
   clf.fit(temp_x_train,y_train)
   dec_val_test=clf.decision_function(temp_x_test)
      
   auc_inc=roc_auc_score(y_test,dec_val_test)
   
   pca_data = numpy.transpose(numpy.vstack((noiserow_test,numpy.array(dec_val_test))))
   pca_vector=pca2.PCA2(pca_data,1)
   min_en2=entropy.entropy_of_vector(pca_vector,y_test)
   
   ch=(auc_inc-auc_base)*1.0
   chen=(min_en2-min_en1)*1.0
   
   if ch>0.0 and chen>0.0:c[0]+=1
   elif ch<0.0 and chen>0.0:c[1]+=1
   elif ch<0.0 and chen<0.0:c[2]+=1
   elif ch>0.0 and chen<0.0:c[3]+=1
   
   tempch=(ch/(1.0 - auc_base))*100.0
   correlation_noise.append(1.0-max_corr)
   change_noise.append(tempch)
   ig_noise.append(chen)

print c[1],c[0]
print c[2],c[3]

print correlation_noise

figpath=path+'/fig/'+dataset
#dx = [0.05,0.2,0.1]
dx=5
#plt.subplot(2,2,1)
plt.axhline(0,linewidth=0.5,ls='--',color="k")
plt.axvline(0,linewidth=0.5,ls='--',color="k")
plt.scatter(change,ig,s=dx,lw = 0.5,facecolor='0.4',alpha=0.8)
#plt.scatter(change_noise,ig_noise,color='k',marker="x")
plt.xlim(xmax=100)
plt.title(title)
plt.xlabel("% Improvement in AUC")
plt.ylabel("Change in minimum entropy")
plt.savefig(figpath+'5.eps', format='eps', dpi=1200)
plt.show()
#plt.subplot(2,2,2)
plt.axvline(0,linewidth=0.5,ls='--',color="k")
plt.scatter(change,correlation,s=dx,lw = 0.1,facecolor='0.65')
plt.scatter(change_noise,correlation_noise,s=10,color='k',marker="x")
plt.xlim(xmax=100)
plt.ylim(ymax=1)
plt.title(title)
plt.xlabel("% Improvement in AUC")
plt.ylabel("1 - |r|")
plt.savefig(figpath+'6.eps', format='eps', dpi=1200)
plt.show()


"""


auc_list=[]
acc_list=[]
if n>1000:
   n=int(0.1*n)
clf=LinearSVC()
for k in range(1,n):
   print k
   tempk=feat[0:k,0]
   topk = [float(integral) for integral in tempk]
   #print topk
   x_train=train_data[:,topk]
   #print x_train.shape
   x_test=test_data[:,topk]
   #print x_test.shape
   clf.fit(x_train,y_train)
   
   #dec_val_test=[]
   
   #for i in range(test_data.shape[0]):
   # dec_val_test.append(clf.decision_function(x_test[i,:])[0])
   
   dec_val_test=clf.decision_function(x_test)
   dlabel=clf.predict(x_test)
   #ptest=clf.predict_proba(x_test)
   
   acc_list.append(1-accuracy_score(y_test,dlabel))
   auc_list.append(roc_auc_score(y_test,dec_val_test))

roc_auc = auc(fpr, tpr) 
roc_auc2 = auc(fpr2, tpr2) 
plt.subplot(2,2,3)
#plt.title('ROC curve (area = %0.2f,%0.2f)' % roc_auc, roc_auc2)
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc,color="k")
plt.plot(fpr2,tpr2,label='ROC curve (area = %0.2f)' % roc_auc,color="k",linestyle='--')
#plt.ylim(ymin=0,ymax=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.subplot(2,2,4) 
plt.plot(range(1,n),acc_list,color="k")
plt.ylim(ymin=0,ymax=1)
plt.xlabel("Top k features")
plt.ylabel("Misclassification Rate")
plt.show()
"""
