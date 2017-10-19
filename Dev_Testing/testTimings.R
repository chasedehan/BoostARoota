#plot timings

library(tidyverse)

setwd("C:/Projects/boostaroota")

timings = read_csv("timings.csv")
timings <- timings %>%
  mutate(TimeInSeconds = (N0 + N1 + N2)/3) %>%
  select(Attributes, Objects, TimeInSeconds)

timings %>%
  ggplot(aes(x=Objects, y = TimeInSeconds, color = factor(Attributes))) +
  geom_smooth(se=FALSE) +
  theme_bw()

timings %>%
  ggplot(aes(x=Attributes, y = TimeInSeconds, color = factor(Objects))) +
  geom_smooth(se=FALSE) +
  theme_bw()

timings %>% 
  arrange(Objects)
