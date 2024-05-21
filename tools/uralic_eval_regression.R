library(dplyr)
library(lme4)
library(lmerTest)

resource_level <- function(language) {
    if (language == "ru" | language == "hu" | language == "fi" | language == "et") {
        "high"
    }
    else {
        "low"
    }
}

finetuning_lines <- list(
    "myv" = 896,
    "sme" = 2001,
    "et" = 5444,
    "fi" = 14981,
    "hu" = 910,
    "ru" = 32768
)

map_finetuning_lines <- function(language) {
    if (language %in% names(finetuning_lines)) {
        finetuning_lines[[language]]
    }
    else {
        0
    }
}

uas_data = read.csv("../output/uralic_uas_data.csv", strip.white=T)
pos_data = read.csv("../output/uralic_pos_data.csv", strip.white=T)

data = merge(uas_data, pos_data, all=T)

data$resource <- as.character(lapply(data$language, resource_level))
data$finetuning_lines <- as.integer(lapply(data$language, map_finetuning_lines))
# set the few-shot finetuning cases to 512
data$finetuning_lines[data$setting == "few-shot"] <- 512
# normalize finetuning lines by units of 512
data$finetuning_lines <- data$finetuning_lines / 512

filtered_data = filter(
    data,
    vocab_alpha == 0.2,
    lapt_alpha >= 0.1,
    vocab_size < 130000,
    lapt_steps > 0
)

filtered_data_a = filter(
    filtered_data,
    lapt_alpha <= 0.2
)

filtered_data_b = filter(
    filtered_data,
    lapt_steps == 100000
)

filtered_data_a$lapt_steps <- filtered_data_a$lapt_steps / min(filtered_data_a$lapt_steps)
filtered_data_a$vocab_size <- filtered_data_a$vocab_size / min(filtered_data_a$vocab_size)
filtered_data_a$lapt_alpha <- filtered_data_a$lapt_alpha / min(filtered_data_a$lapt_alpha)

filtered_data_b$lapt_steps <- filtered_data_b$lapt_steps / min(filtered_data_b$lapt_steps)
filtered_data_b$vocab_size <- filtered_data_b$vocab_size / min(filtered_data_b$vocab_size)
filtered_data_b$lapt_alpha <- filtered_data_b$lapt_alpha / min(filtered_data_b$lapt_alpha)

######## ANOVA OF TASK-SLOPE INTERACTIONS ########

finetuned_task_fit = lmer(
    accuracy ~ lapt_steps*task + vocab_size*task + lapt_alpha*task + finetuning_lines + (1 | language),
    data=filter(filtered_data_a, setting == 'few-shot' | setting == 'full-train')
)
#print(summary(finetuned_task_fit))
#print(coef(finetuned_task_fit))

finetuned_task_control_fit = lmer(
    accuracy ~ lapt_steps + vocab_size + lapt_alpha + finetuning_lines + task + (1 | language),
    data=filter(filtered_data_a, setting == 'few-shot' | setting == 'full-train')
)
#print(summary(finetuned_task_control_fit))
#print(coef(finetuned_task_control_fit))

print("######## FINETUNED TASK INTERACTION ANOVA ########")
print(anova(finetuned_task_fit, finetuned_task_control_fit))

zeroshot_task_fit = lmer(
    accuracy ~ lapt_steps*task + vocab_size*task + lapt_alpha*task + (1 | language),
    data=filter(filtered_data_a, setting == 'zero-shot')
)

zeroshot_task_control_fit = lmer(
    accuracy ~ lapt_steps + vocab_size + lapt_alpha + task + (1 | language),
    data=filter(filtered_data_a, setting == 'zero-shot')
)

print("######## ZEROSHOT TASK INTERACTION ANOVA ########")
print(anova(zeroshot_task_fit, zeroshot_task_control_fit))

######## ANOVA OF STEPS-VOCAB INTERACTION ########

interactions_fit = lmer(
    accuracy ~ lapt_steps*vocab_size + resource:lapt_alpha + finetuning_lines + task + (1 | language),
    data=filter(filtered_data_a, setting == 'few-shot' | setting == 'full-train')
)
#print(summary(interactions_fit))
#print(coef(interactions_fit))

interactions_control_fit = lmer(
    accuracy ~ lapt_steps + vocab_size + finetuning_lines + task + resource:lapt_alpha + (1 | language),
    data=filter(filtered_data_a, setting == 'few-shot' | setting == 'full-train')
)

print("######## STEPS-VOCAB INTERACTION ANOVA ########")
print(anova(interactions_fit, interactions_control_fit))

######## MAIN REGRESSIONS ########

print("######## FINETUNED REGRESSION A ########")
finetuned_fit_a = lmer(
    accuracy ~ lapt_steps + vocab_size + finetuning_lines + task + resource:lapt_alpha + (1 | language),
    data=filter(filtered_data_a, setting == 'few-shot' | setting == 'full-train')
)
print(summary(finetuned_fit_a))
print(coef(finetuned_fit_a))

print("######## FINETUNED REGRESSION B ########")
finetuned_fit_b = lmer(
    accuracy ~ vocab_size + finetuning_lines + task + resource:lapt_alpha + (1 | language),
    data=filter(filtered_data_b, setting == 'few-shot' | setting == 'full-train')
)
print(summary(finetuned_fit_b))
print(coef(finetuned_fit_b))


print("######## ZEROSHOT REGRESSION A ########")
zeroshot_fit_a = lmer(
    accuracy ~ lapt_steps + vocab_size + lapt_alpha + task + (1 | language),
    data=filter(filtered_data_a, setting == 'zero-shot')
)
print(summary(zeroshot_fit_a))
print(coef(zeroshot_fit_a))

print("######## ZEROSHOT REGRESSION B ########")
zeroshot_fit_b = lmer(
    accuracy ~ vocab_size + lapt_alpha + task + (1 | language),
    data=filter(filtered_data_b, setting == 'zero-shot')
)
print(summary(zeroshot_fit_b))
print(coef(zeroshot_fit_b))
