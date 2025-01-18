#!/bin/bash  
PATH=/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/opt/tcl-tk/bin:/Users/ahemf/opt/anaconda3/bin:/Users/ahemf/opt/anaconda3/condabin:/usr/local/opt/openssl@1.1/bin:/Users/ahemf/bin:/Users/ahemf/.local/bin:/Users/ahemf/.toolbox/bin:/usr/local/bin:/Users/ahemf/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/usr/local/bin:$PATH
# latexmk -C
bibtex main  
pdflatex -interaction=nonstopmode main  
bibtex main  
pdflatex -interaction=nonstopmode main  
