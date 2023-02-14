cdata<-read.csv("U:/computer/Python BERT/CFA data/outQ5top13_2_reformatted.csv") #using larger validation set
dim(cdata)


head(cdata)

check<-read.csv("U:/computer/Python BERT/CFA data/NAXION_Q5_top13tags_for_BERT.csv")
table(check$Tag)
#2   3   4   6   7   8  12  13  52  53  54  69  70 
#221 135 102 114  91  85 101 275 152 300 157  85 157 

dim(check)
#[1] 1975    3

check2<-read.csv("U:/computer/Python BERT/CFA data/NAXION_Q5_top13text_for_BERT.csv")
dim(check2)
#[1] 1378    3


#actual codes
all_acodes<-c(cdata$acode1,cdata$acode2,cdata$acode3,cdata$acode4,cdata$acode5,cdata$acode6,cdata$acode7,cdata$acode8,cdata$acode9,cdata$acode10)
sort(table(all_acodes))
length(unique(all_acodes))
unique(all_acodes) #includes NA
sum(!is.na(all_acodes))
table(all_acodes)
#2   3   4   6   7   8  12  13  52  53  54  69  70 
#221 135 102 114  91  85 101 275 152 300 157  85 157 

#predicted codes
all_pcodes<-c(cdata$pcode1,cdata$pcode2,cdata$pcode3,cdata$pcode4,cdata$pcode5,cdata$pcode6,cdata$pcode7,cdata$pcode8, cdata$pcode9, cdata$pcode10,cdata$pcode11,cdata$pcode12,cdata$pcode13,cdata$pcode14)
table(all_pcodes)
length(unique(all_pcodes))
unique(all_pcodes) #includes NA
#[1]  2 54 NA 53 13  6 70  3 52 69  7

#all_pcodes v1 validation set .3
#2   3    6   7  13  52  53  54  69  70 
#205 102  72  55 252 163 296 111  18 162 

#all_pcodes v2 validation set .5
# 2   3      7  13  52  53   54  69  70 
#206  28     30 192 150 302  14   4 137 

#cc<-c(8,69,7,12,4,6,3,52,54,70,2,13,53)

######################################################################################################################################################
######################################################################################################################################################

true_8<-rep(0,nrow(cdata))
true_69<-rep(0,nrow(cdata))
true_7<-rep(0,nrow(cdata))
true_12<-rep(0,nrow(cdata))
true_4<-rep(0,nrow(cdata))
true_6<-rep(0,nrow(cdata))
true_3<-rep(0,nrow(cdata))
true_52<-rep(0,nrow(cdata))
true_54<-rep(0,nrow(cdata))
true_70<-rep(0,nrow(cdata))
true_2<-rep(0,nrow(cdata))
true_13<-rep(0,nrow(cdata))
true_53<-rep(0,nrow(cdata))






true_8[(cdata$acode1==8)|(cdata$acode2==8)|(cdata$acode3==8)|(cdata$acode4==8)
         |(cdata$acode5==8)|(cdata$acode6==8)|(cdata$acode7==8)|(cdata$acode8==8)|
         (cdata$acode9==8)|(cdata$acode10==8)]<-1

true_69[(cdata$acode1==69)|(cdata$acode2==69)|(cdata$acode3==69)|(cdata$acode4==69)
        |(cdata$acode5==69)|(cdata$acode6==69)|(cdata$acode7==69)|(cdata$acode8==69)|
          (cdata$acode9==69)|(cdata$acode10==69)]<-1

true_7[(cdata$acode1==7)|(cdata$acode2==7)|(cdata$acode3==7)|(cdata$acode4==7)
       |(cdata$acode5==7)|(cdata$acode6==7)|(cdata$acode7==7)|(cdata$acode8==7)|
               (cdata$acode9==7)|(cdata$acode10==7)]<-1

true_12[(cdata$acode1==12)|(cdata$acode2==12)|(cdata$acode3==12)|(cdata$acode4==12)
       |(cdata$acode5==12)|(cdata$acode6==12)|(cdata$acode7==12)|(cdata$acode8==12)|
               (cdata$acode9==12)|(cdata$acode10==12)]<-1

true_4[(cdata$acode1==4)|(cdata$acode2==4)|(cdata$acode3==4)|(cdata$acode4==4)
       |(cdata$acode5==4)|(cdata$acode6==4)|(cdata$acode7==4)|(cdata$acode8==4)|
               (cdata$acode9==4)|(cdata$acode10==4)]<-1

true_6[(cdata$acode1==6)|(cdata$acode2==6)|(cdata$acode3==6)|(cdata$acode4==6)
       |(cdata$acode5==6)|(cdata$acode6==6)|(cdata$acode7==6)|(cdata$acode8==6)|
               (cdata$acode9==6)|(cdata$acode10==6)]<-1

true_3[(cdata$acode1==3)|(cdata$acode2==3)|(cdata$acode3==3)|(cdata$acode4==3)
       |(cdata$acode5==3)|(cdata$acode6==3)|(cdata$acode7==3)|(cdata$acode8==3)|
               (cdata$acode9==3)|(cdata$acode10==3)]<-1

true_52[(cdata$acode1==52)|(cdata$acode2==52)|(cdata$acode3==52)|(cdata$acode4==52)
       |(cdata$acode5==52)|(cdata$acode6==52)|(cdata$acode7==52)|(cdata$acode8==52)|
               (cdata$acode9==52)|(cdata$acode10==52)]<-1

true_54[(cdata$acode1==54)|(cdata$acode2==54)|(cdata$acode3==54)|(cdata$acode4==54)
       |(cdata$acode5==54)|(cdata$acode6==54)|(cdata$acode7==54)|(cdata$acode8==54)|
               (cdata$acode9==54)|(cdata$acode10==54)]<-1

true_70[(cdata$acode1==70)|(cdata$acode2==70)|(cdata$acode3==70)|(cdata$acode4==70)
       |(cdata$acode5==70)|(cdata$acode6==70)|(cdata$acode7==70)|(cdata$acode8==70)|
               (cdata$acode9==70)|(cdata$acode10==70)]<-1

true_2[(cdata$acode1==2)|(cdata$acode2==2)|(cdata$acode3==2)|(cdata$acode4==2)
       |(cdata$acode5==2)|(cdata$acode6==2)|(cdata$acode7==2)|(cdata$acode8==2)|
               (cdata$acode9==2)|(cdata$acode10==2)]<-1

true_13[(cdata$acode1==13)|(cdata$acode2==13)|(cdata$acode3==13)|(cdata$acode4==13)
       |(cdata$acode5==13)|(cdata$acode6==13)|(cdata$acode7==13)|(cdata$acode8==13)|
               (cdata$acode9==13)|(cdata$acode10==13)]<-1

true_53[(cdata$acode1==53)|(cdata$acode2==53)|(cdata$acode3==53)|(cdata$acode4==53)
       |(cdata$acode5==53)|(cdata$acode6==53)|(cdata$acode7==53)|(cdata$acode8==53)|
               (cdata$acode9==53)|(cdata$acode10==53)]<-1



#cc<-c(8,69,7,12,4,6,3,52,54,70,2,13,53)

pred_8<-rep(0,nrow(cdata))
pred_69<-rep(0,nrow(cdata))
pred_7<-rep(0,nrow(cdata))
pred_12<-rep(0,nrow(cdata))
pred_4<-rep(0,nrow(cdata))
pred_6<-rep(0,nrow(cdata))
pred_3<-rep(0,nrow(cdata))
pred_52<-rep(0,nrow(cdata))
pred_54<-rep(0,nrow(cdata))
pred_70<-rep(0,nrow(cdata))
pred_2<-rep(0,nrow(cdata))
pred_13<-rep(0,nrow(cdata))
pred_53<-rep(0,nrow(cdata))



pred_8[(cdata$pcode1==8)|(cdata$pcode2==8)|(cdata$pcode3==8)|(cdata$pcode4==8)
       |(cdata$pcode5==8)|(cdata$pcode6==8)|(cdata$pcode7==8)|(cdata$pcode8==8)|
               (cdata$pcode9==8)|(cdata$pcode10==8)]<-1

pred_69[(cdata$pcode1==69)|(cdata$pcode2==69)|(cdata$pcode3==69)|(cdata$pcode4==69)
       |(cdata$pcode5==69)|(cdata$pcode6==69)|(cdata$pcode7==69)|(cdata$pcode8==69)|
               (cdata$pcode9==69)|(cdata$pcode10==69)]<-1

pred_7[(cdata$pcode1==7)|(cdata$pcode2==7)|(cdata$pcode3==7)|(cdata$pcode4==7)
       |(cdata$pcode5==7)|(cdata$pcode6==7)|(cdata$pcode7==7)|(cdata$pcode8==7)|
               (cdata$pcode9==7)|(cdata$pcode10==7)]<-1

pred_12[(cdata$pcode1==12)|(cdata$pcode2==12)|(cdata$pcode3==12)|(cdata$pcode4==12)
       |(cdata$pcode5==12)|(cdata$pcode6==12)|(cdata$pcode7==12)|(cdata$pcode8==12)|
               (cdata$pcode9==12)|(cdata$pcode10==12)]<-1

pred_4[(cdata$pcode1==4)|(cdata$pcode2==4)|(cdata$pcode3==4)|(cdata$pcode4==4)
       |(cdata$pcode5==4)|(cdata$pcode6==4)|(cdata$pcode7==4)|(cdata$pcode8==4)|
               (cdata$pcode9==4)|(cdata$pcode10==4)]<-1

pred_6[(cdata$pcode1==6)|(cdata$pcode2==6)|(cdata$pcode3==6)|(cdata$pcode4==6)
       |(cdata$pcode5==6)|(cdata$pcode6==6)|(cdata$pcode7==6)|(cdata$pcode8==6)|
               (cdata$pcode9==6)|(cdata$pcode10==6)]<-1

pred_3[(cdata$pcode1==3)|(cdata$pcode2==3)|(cdata$pcode3==3)|(cdata$pcode4==3)
       |(cdata$pcode5==3)|(cdata$pcode6==3)|(cdata$pcode7==3)|(cdata$pcode8==3)|
               (cdata$pcode9==3)|(cdata$pcode10==3)]<-1

pred_52[(cdata$pcode1==52)|(cdata$pcode2==52)|(cdata$pcode3==52)|(cdata$pcode4==52)
       |(cdata$pcode5==52)|(cdata$pcode6==52)|(cdata$pcode7==52)|(cdata$pcode8==52)|
               (cdata$pcode9==52)|(cdata$pcode10==52)]<-1

pred_54[(cdata$pcode1==54)|(cdata$pcode2==54)|(cdata$pcode3==54)|(cdata$pcode4==54)
       |(cdata$pcode5==54)|(cdata$pcode6==54)|(cdata$pcode7==54)|(cdata$pcode8==54)|
               (cdata$pcode9==54)|(cdata$pcode10==54)]<-1

pred_70[(cdata$pcode1==70)|(cdata$pcode2==70)|(cdata$pcode3==70)|(cdata$pcode4==70)
        |(cdata$pcode5==70)|(cdata$pcode6==70)|(cdata$pcode7==70)|(cdata$pcode8==70)|
                (cdata$pcode9==70)|(cdata$pcode10==70)]<-1

pred_2[(cdata$pcode1==2)|(cdata$pcode2==2)|(cdata$pcode3==2)|(cdata$pcode4==2)
        |(cdata$pcode5==2)|(cdata$pcode6==2)|(cdata$pcode7==2)|(cdata$pcode8==2)|
                (cdata$pcode9==2)|(cdata$pcode10==2)]<-1

pred_13[(cdata$pcode1==13)|(cdata$pcode2==13)|(cdata$pcode3==13)|(cdata$pcode4==13)
        |(cdata$pcode5==13)|(cdata$pcode6==13)|(cdata$pcode7==13)|(cdata$pcode8==13)|
                (cdata$pcode9==13)|(cdata$pcode10==13)]<-1

pred_53[(cdata$pcode1==53)|(cdata$pcode2==53)|(cdata$pcode3==53)|(cdata$pcode4==53)
        |(cdata$pcode5==53)|(cdata$pcode6==53)|(cdata$pcode7==53)|(cdata$pcode8==53)|
                (cdata$pcode9==53)|(cdata$pcode10==53)]<-1



######################################################################################################################################################
######################################################################################################################################################
cc<-c(8,69,7,12,4,6,3,52,54,70,2,13,53)

mytrue<-cbind(true_8,true_69,true_7,true_12,true_4,true_6,true_3,true_52,true_54,true_70,true_2,true_13,true_53)
mypred<-cbind(pred_8,pred_69,pred_7,pred_12,pred_4,pred_6,pred_3,pred_52,pred_54,pred_70,pred_2,pred_13,pred_53)

colSums(mytrue)
colSums(mypred)

# > colSums(mytrue)
# true_8 true_69  true_7 true_12  true_4  true_6  true_3 true_52 true_54 true_70  true_2 true_13 true_53 
# 85      85      91     101     102     114     135     152     157     157     221     275     300 
# > colSums(mypred)
#pred_8 pred_69  pred_7 pred_12  pred_4  pred_6  pred_3 pred_52 pred_54 pred_70  pred_2 pred_13 pred_53 
#0       4      30       0       0       0      28     150      14     137     206     192     302 

#nit<-length(unique(all_acodes))-1 #subtract 1 because includes NA
nit<-13

recall<-rep(0,nit)
precision<-rep(0,nit)
f1<-rep(0,nit)



for (i in (1:nit)){




## Calculating Recall
## What proportion of actual positives were identified?
### Recall true_positive/(true_positive+false_negative)

recall[i]<-sum(mypred[mytrue[,i]==1,i])/(sum(mypred[mytrue[,i]==1,i])+sum(mytrue[mypred[,i]==0,i]))


## Calculating Precision
## Of the scored positives, what proportion is correct?
### Precision true_positive/(true_positive+false_positive)

precision[i]<-sum(mypred[mytrue[,i]==1,i])/(sum(mypred[mytrue[,i]==1,i])+sum(mypred[mytrue[,i]==0,i]))



f1[i]<-2*(precision[i]*recall[i])/(precision[i]+recall[i])




}



xx<-cbind(cc,
       recall,
precision,
f1, 0, 0)

for (i in 1:13){
        xx[i,5]<-sum(mytrue[,i])
        xx[i,6]<-sum(mypred[,i])
}

dimnames(xx)[[2]]<-c("code","recall","precision","f1", "True n","Pred n")

round(xx,3)

code recall precision    f1 True n Pred n
[1,]    8  0.000       NaN   NaN     85      0
[2,]   69  0.047     1.000 0.090     85      4
[3,]    7  0.330     1.000 0.496     91     30
[4,]   12  0.000       NaN   NaN    101      0
[5,]    4  0.000       NaN   NaN    102      0
[6,]    6  0.000       NaN   NaN    114      0
[7,]    3  0.207     1.000 0.344    135     28
[8,]   52  0.901     0.913 0.907    152    150
[9,]   54  0.089     1.000 0.164    157     14
[10,]   70  0.822     0.942 0.878    157    137
[11,]    2  0.873     0.937 0.904    221    206
[12,]   13  0.625     0.896 0.737    275    192
[13,]   53  0.937     0.930 0.934    300    302

code recall precision    f1 True n Pred n
[1,]    8  0.000       NaN   NaN     85      0
[2,]   69  0.212     1.000 0.350     85     18
[3,]    7  0.604     1.000 0.753     91     55
[4,]   12  0.000       NaN   NaN    101      0
[5,]    4  0.000       NaN   NaN    102      0
[6,]    6  0.623     0.986 0.763    114     72
[7,]    3  0.748     0.990 0.852    135    102
[8,]   52  0.947     0.883 0.914    152    163
[9,]   54  0.694     0.982 0.813    157    111
[10,]   70  0.949     0.920 0.934    157    162
[11,]    2  0.900     0.971 0.934    221    205
[12,]   13  0.811     0.885 0.846    275    252
[13,]   53  0.943     0.956 0.950    300    296

# 
# 
# aa<-sort(table(all_acodes))
# aa
# 11   3   2   6   4   5   8   7  12  14  13   1   9 
# 5  79 138 282 287 321 323 414 445 518 523 574 652 
# 
# bb<-sort(table(all_pcodes))
# bb
# 3   8   6   5   4  12   7  14   1  13   9 
# 2  87 150 198 212 254 306 411 445 672 791 
# > 
# 
#         
# table(true_83) 
# table()
#         
# table(true_83) 
# table(pred_19[true_83==1])
# table(pred_5[true_83==1])
# table(pred_64[true_83==1])
# table(pred_86[true_83==1])
# table(pred_54[true_83==1])
# table(pred_83[true_83==1])
# table(pred_10[true_83==1])
# table(pred_81[true_83==1])
# table(pred_22[true_83==1])
# table(pred_2[true_83==1])
# table(pred_82[true_83==1])
# table(pred_17[true_83==1])
# table(pred_43[true_83==1])
# 
# 
# temp<-cbind(cdata$Body[true_83==1],cdata$pcode1[true_83==1],cdata$pcode2[true_83==1],cdata$pcode3[true_83==1],cdata$pcode4[true_83==1],cdata$pcode5[true_83==1],cdata$pcode6[true_83==1],cdata$pcode10[true_83==1])
# write.csv(temp,file="U:/computer/Python BERT/CFA data/Q1_code83.csv")
#         
######################################################################################################################################################
######################################################################################################################################################

