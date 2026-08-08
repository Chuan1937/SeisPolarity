SPHINXBUILD   ?= sphinx-build
SPHINXOPTS    ?=
BUILDDIR      = _build

.PHONY: help clean html en zh_CN docs

help:
	@echo "Targets: html (build en + zh_CN), en, zh_CN, clean"

docs: html

html: en zh_CN

en:
	$(SPHINXBUILD) -b html en $(BUILDDIR)/html $(SPHINXOPTS) $(O)

zh_CN:
	$(SPHINXBUILD) -b html zh_CN $(BUILDDIR)/html/zh_CN $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)
