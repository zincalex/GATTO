#The file will be:
#  - Cora_0.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - Cora_1:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - CiteSeer_0.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - CiteSeer_1.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col

ALPHA <- 0.05

data_Cora_0 <- read.csv("Cora_0.csv")
data_Cora_1 <- read.csv("Cora_1.csvv")
data_CiteSeer_0 <- read.csv("CiteSeer_0.csv")
data_CiteSeer_1 <- read.csv("CiteSeer_1.csv")
cols <- c("Acc","Prec","Rec","F1S")

print("----Cora ANALYSIS----")
#Cora part
for (col in cols){
    
    cora_col_0 <- data_Cora_0[[col]]
    cora_col_1 <- data_Cora_1[[col]]

    print(col,"of Cora_0: ",mean(cora_col_0),var(cora_col_0))
    print(col,"of Cora_1: ",mean(cora_col_1),var(cora_col_1))

    s_p-value_0 <- shapiro.test(cora_col_0)$p.value 
    s_p-value_1 <- shapiro.test(cora_col_1)$p.value

    if (s_p-value_0 >= ALPHA && s_p-value_1 >= ALPHA){
        #Statistically as normal

        tt_pval <- t.test(cora_col_0, cora_col_1, var.equal = TRUE)$p.value  # Equal variance assumption
        if (tt_pval >= ALPHA){
            print(col,"same behaviour! (X's as N(µ,ß1), Y's as N(µ,ß1))",tt_pval)
        }else{
            print(col,"statistically significant difference! (X's as N(µ,ß1), Y's as N(µ,ß1))",tt_pval)
        }

        tf_pval <- t.test(cora_col_0, cora_col_1, var.equal = FALSE)$p.value
        if (tf_pval >= ALPHA){
            print(col,"same behaviour! (X's as N(µ,ß1), Y's as N(µ,ß2))",tf_pval)
        }else{
            print(col,"statistically significant difference! (X's as N(µ,ß1), Y's as N(µ,ß2))",tf_pval)
        }
    }else{
        #Not statistically normal
        print("NO DIS AS NORMAL..")
        mw_pval <- wilcox.test(cora_col_0, cora_col_1)$p.value
        if (mw_pval >= ALPHA){
            print(col,"same behaviour!",mw_pval)
        }else{
            print(col,"statistically significant difference!",mw_pval)
        }
    }
}

print("----CiteSeer ANALYSIS----")

#CiteSeer part
for (col in cols){
    
    cseer_col_0 <- data_CiteSeer_0[[col]]
    cseer_col_1 <- data_CiteSeer_1[[col]]

    print(col,"of CiteSeer_0: ",mean(cseer_col_0),var(cseer_col_0))
    print(col,"of CiteSeer_1: ",mean(cseer_col_1),var(cseer_col_1))

    s_p-value_0 <- shapiro.test(cseer_col_0)$p.value 
    s_p-value_1 <- shapiro.test(cseer_col_1)$p.value

    if (s_p-value_0 >= ALPHA && s_p-value_1 >= ALPHA){
        #Statistically as normal

        tt_pval <- t.test(cseer_col_0, cseer_col_1, var.equal = TRUE)$p.value  # Equal variance assumption
        if (tt_pval >= ALPHA){
            print(col,"same behaviour! (X's as N(µ,ß1), Y's as N(µ,ß1))",tt_pval)
        }else{
            print(col,"statistically significant difference! (X's as N(µ,ß1), Y's as N(µ,ß1))",tt_pval)
        }

        tf_pval <- t.test(cseer_col_0, cseer_col_1, var.equal = FALSE)$p.value
        if (tf_pval >= ALPHA){
            print(col,"same behaviour! (X's as N(µ,ß1), Y's as N(µ,ß2))",tf_pval)
        }else{
            print(col,"statistically significant difference! (X's as N(µ,ß1), Y's as N(µ,ß2))",tf_pval)
        }
    }else{
        #Not statistically normal
        print("NO DIS AS NORMAL..")
        mw_pval <- wilcox.test(cseer_col_0, cseer_col_1)$p.value
        if (mw_pval >= ALPHA){
            print(col,"same behaviour!",mw_pval)
        }else{
            print(col,"statistically significant difference!",mw_pval)
        }
    }
}