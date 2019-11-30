LATEX=latexmk -pdf -pvc -shell-escape
#LATEX=pdflatex
BIBTEX=bibtex
Latex = $(wildcard *.tex)
Pdfs = $(Latex:.tex=.pdf)

all: $(Pdfs)

%.pdf: %.tex
				$(LATEX) $<
				#$(BIBTEX) $(<:.tex=) #Uncomment to add bibtex to latexpdf
				#$(LATEX) $ $< #Uncomment to add bibtex to latexpdf
				open $@ # Open pdf in osx, otherwise remove


clean:
				rm -rf *.fls *.synctex.gz *.pdf *.log *.aux *.bbl *.blg *.dvi *.fdb_latexmk
