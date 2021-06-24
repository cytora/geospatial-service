.PHONY : helm test all

all:
	make lint
	make test
	make helm

helm:
	cd helm/${SERVICE} && \
	sed -i -e "s/CIRCLE_BUILD_NUM/${CIRCLE_BUILD_NUM}/g" Chart.yaml && \
	sed -i -e "s/CIRCLE_BUILD_NUM/${CIRCLE_BUILD_NUM}/g" values.yaml && \
	sed -i -e "s/SERVICE/${SERVICE}/g" values.yaml && \
	sed -i -e "s/SERVICE/${SERVICE}/g" Chart.yaml && \
	sed -i -e "s/SERVICE/${SERVICE}/g" release-name && \
	sed -i -e "s/SERVICE/${SERVICE}/g" test-release-name && \
	cd ../ && \
	helm lint ./${SERVICE} && \
	helm package ./${SERVICE}

install:
	bash scripts/build.sh install

lint:
	bash scripts/build.sh lint

test:
	bash scripts/build.sh test

container:
	bash scripts/build.sh container

local-run:
	bash scripts/local-run.sh

component-test:
	pytest component-tests
