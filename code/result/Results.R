#The file will be:
#  - Cora_0.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - Cora_1:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - CiteSeer_0.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col
#  - CiteSeer_1.csv:
#     "Acc","Prec","Rec","F1S" -- 10 values x col

setwd("code/result")
sink("report.txt")
ALPHA <- 0.05

cols <- c("Acc","Prec","Rec","F1S")

for(data in c("Cora","CiteSeer")){

    print(paste("---",data,"Analysis ---"))
    filename_0 <- paste(data,"_0.csv",sep="")
    filename_1 <- paste(data,"_1.csv",sep="")

    data_0 <- read.csv(filename_0)
    data_1 <- read.csv(filename_1)

    for (col in cols){
        
        col_0 <- data_0[[col]]
        col_1 <- data_1[[col]]

        print(paste(col,data,"0 --","mean:",mean(col_0),"- var:",var(col_0)))
        print(paste(col,data,"1 --","mean:",mean(col_1),"- var:",var(col_1)))

        s_p_value_0 <- shapiro.test(col_0)$p.value 
        s_p_value_1 <- shapiro.test(col_1)$p.value

        if (s_p_value_0 >= ALPHA && s_p_value_1 >= ALPHA){
            #Statistically as normal

            tt_pval <- t.test(col_0, col_1, var.equal = TRUE)$p.value  # Equal variance assumption
            if (tt_pval >= ALPHA){
                print(paste(col,"have same behaviour! [Xi ~ N(µ,s1) and Y i~ N(µ,s1)] -- p-value:",tt_pval))
            }else{
                print(paste(col,"statistically significant difference! [Xi ~ N(µ,s1) and Y i~ N(µ,s1)] -- p-value: ",tt_pval))
            }

            tf_pval <- t.test(col_0, col_1, var.equal = FALSE)$p.value
            if (tf_pval >= ALPHA){
                print(paste(col,"have same behaviour! [Xi ~ N(µ,s1) and Yi ~ N(µ,s2)] -- p-value:",tf_pval))
            }else{
                print(paste(col,"statistically significant difference! [Xi ~ N(µ,s1) and Yi ~ N(µ,s2)] -- p-value:",tf_pval))
            }
        }else{
            #Not statistically normal
            print("!!! NO ~ N !!!")
            mw_pval <- wilcox.test(col_0, col_1,exact = FALSE)$p.value
            if (mw_pval >= ALPHA){
                print(paste(col,"same behaviour! -- p-value:",mw_pval))
            }else{
                print(paste(col,"statistically significant difference! -- p-value:",mw_pval))
            }
        }
    }
}