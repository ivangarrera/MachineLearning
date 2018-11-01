pdflatex -synctex=1 -interaction=nonstopmode main.tex > NUL

pdflatex -synctex=1 -interaction=nonstopmode main.tex > NUL

DEL main.aux main.synctex.gz main.toc main.log main.blg main.out main.bbl

explorer.exe "main.pdf"
