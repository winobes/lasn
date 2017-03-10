library(car)
data = read.csv("chg_e_central_y2y.csv", header = TRUE)
attach(data)
scatterplot(e_central_chg ~ coord_given)

