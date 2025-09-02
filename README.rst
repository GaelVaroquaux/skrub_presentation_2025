skrub for tables: Less wrangling, more machine learning
========================================================

Video of the presentation: https://www.youtube.com/watch?v=hdWWhwmRpbA

This code currently requires the following PR https://github.com/skrub-data/skrub/pull/1233 soon to be merged in skrub

Presentation at Python Exchange in Feb 2025
---------------------------------------------

While tabular data is central to all organizations, it seems left out of the AI discussion, which has focused on images, text and sound. Ineed, for data science, most of the excitement is in machine learning, but most of the work happens before. Tables often require extensive manual transformation or "data wrangling".

I will discuss how we progressively rethought this process, building machine learning tool that require less wrangling. We are building a new library, skrub (https://skrub-data.org), that facilitates complex tabular-learning pipelines, writing as much as possible wrangling as high-level operations and automating them. A few lines of skrub can spare you dozens of wrangling lines!

