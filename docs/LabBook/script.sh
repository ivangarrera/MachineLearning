#!/bin/bash


pdflatex -synctex=1 -interaction=nonstopmode main.tex 1,2> /dev/null

pdflatex -synctex=1 -interaction=nonstopmode main.tex 1,2> /dev/null

rm main.aux main.log main.synctex.gz main.toc main.blg main.out main.bbl

evince main.pdf
