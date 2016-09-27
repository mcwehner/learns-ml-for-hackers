data	:= Nottingham

ai_composer: $(data)
	python $@.py

clean:
	rm -rf $(data)

$(data):
	curl -O http://www-etud.iro.umontreal.ca/~boulanni/$@.zip
	unzip $@.zip
	rm $@.zip

.PHONY: ai_composer clean
