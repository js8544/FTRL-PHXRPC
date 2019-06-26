#include "LR.h"
#include "corpus.h"
#include "sparse_vector.h"
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <omp.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <map>
// #include <omp.h>
// #include <pthread.h>
using namespace std;

#define SHM_KEY 0x1234

struct shmseg {
	int mode; //-1:waiting, 0: user_id ready, 1: ad_id ready, 2:feedback ready
   	long long user_id;
   	long long ad_id;
   	bool feedback; 
};

inline int sgn(double a){
	return (0<a)-(a>0);
}

class FTRL
{
private:
	//Number of Features
	int d;

	//Parameters
	double alpha;
	double beta;
	double l1;
	double l2;

	//Decision Function
	LR lr;

	//Model
	SpVec w;
	SpVec z;
	SpVec n;

public:
	FTRL():alpha(0.5),beta(1.0),l1(1.0),l2(1.0),d(4){};

	FTRL(int D, double Alpha, double Beta, double L1, double L2):d(D),alpha(Alpha),beta(Beta),l1(L1),l2(L2){};

	double perdict(SpVec& x){
		return lr.decision(w,x);
	}

	double update(SpVec& x, double y){
		#pragma omp parallel for
		for(int i = 0;i < d;++i){
			if(abs(z.get_value(i)) > l1){
				#pragma omp critical
				w.set_value(i,-1.0/((beta+sqrt(n.get_value(i)))/alpha+l2)*(z.get_value(i)-sgn(z.get_value(i))*l1));
			}
			else{
				#pragma omp critical
				w.set_value(i,0);
			}
		}
		double p = perdict(x);
		SpVec g = lr.gradient(p,y,x);

		#pragma omp parallel for
		for(int i = 0;i < d;i++){
			double temp_g = (p-y)*(x.get_value(i));
			g.set_value(i, temp_g);
			double sigma = (1.0/alpha)*(sqrt(n.get_value(i)+(temp_g*temp_g))-sqrt(n.get_value(i)));
			double temp_z = z.get_value(i)+temp_g-sigma*w.get_value(i);
			double temp_n = n.get_value(i)+temp_g*temp_g;
			
			#pragma omp critical
			z.set_value(i,temp_z);
			#pragma omp critical
			n.set_value(i,temp_n);
		}

		return lr.loss(p,y);
	}

	void train(corpus& data){
		cout<<"training "<<data.size()<<" data"<<endl;
		double correct = 0;
		double wrong = 0;

		for(int i=0;i<data.size();i++){
			cout<<"data "<<i<<" ";
			int p = (perdict(data[i].x)>0.5);
			double loss = update(data[i].x,data[i].y);

			if(p==data[i].y){
				correct++;
			}
			else{
				wrong++;
			}
			cout<<"loss: "<< loss <<" ";
			cout<<"accuracy: "<<correct/(correct+wrong)<<endl;



		}
		cout<<"trained weight:"<<endl;
		w.print_value();
	}

	double test(corpus& data){
		cout<<"testing "<<data.size()<<" data"<<endl;
		double correct = 0;
		double wrong = 0;

		#pragma omp parallel for
		for(int i=0;i<data.size();i++){

			// cout<<"data "<<i<<":"<<perdict(data[i].x)<<"--"<<data[i].y<<endl;
			int p=0;
			if(perdict(data[i].x)>0.5){
				p = 1;
			}
			if(p==data[i].y){
				correct++;
			}
			else{
				wrong++;
			}
		}
		return correct/(correct+wrong);
	}

	void save(ofstream* FILE){
		sp_iter it = w.vc.begin();
		while(it!=w.vc.end()){
			(*FILE)<<it->first<<": "<<it->second<<endl;
			it++;
		}
	}

	void load(ifstream* FILE){
		string line;
		while(getline(*FILE,line)){
			vector<string> temp = parse_feature(line,":");
			
			long x = stoi(temp[0]);
			double v = stod(temp[1]);
			w.set_value(x,v);
		}
	}
};

int main(int argc, char const *argv[])
{
	ifstream FILE;
	FILE.open(argv[1]);
	cout<<"Loading train set\n";
	corpus train_set(&FILE);
	

	FTRL ftrl(train_set.d, 0.5, 1, 1, 1);

	cout<<"Loading Model\n";
	ifstream MODEL;
	MODEL.open(argv[2]);
	ftrl.load(&MODEL);


	cout<<"accuracy: "<<ftrl.test(train_set)<<endl;
	

	int shmid;
	shmid = shmget(SHM_KEY, sizeof(struct shmseg), 0644|IPC_CREAT);
	if (shmid == -1) {
	  perror("Shared memory");
	  return 1;
	}

	shmseg *shmp;

	shmp = (struct shmseg*) shmat(shmid, NULL, 0);
	if (shmp == (void *) -1) {
		perror("Shared memory attach");
		return 1;
	}

	shmp->mode = -1;

	int cur = 0;
	while(1){
		if(shmp->mode==0){
			cout<<"receive user id "<<shmp->user_id<<"\n";

			// sparse_vector user = get_user_feature(shmp->user_id);
			// map<int,sparse_vector> ad = get_ads();

			double res = 0;
			int ad_id;
			for(int i=0;i<5;i++){
				double p = ftrl.perdict(train_set[cur+i].x);
				cout<<"prediction: "<<cur+i<<" : "<<p<<"\n";
				if(p>res){
					res = p;
					ad_id = cur+i;
				}
			}
			cur+=5;
			shmp->ad_id = ad_id;

			cout<<"sent ad_id "<<shmp->ad_id<<"\n";
			shmp->mode = 1;
		}

		if(shmp->mode==2){
			// sparse_vector ad = get_ad(shmp->ad_id);

			cout<<"receive ad id "<<shmp->ad_id<<"\n";

			sparse_vector ad = train_set[shmp->ad_id].x;

			cout<<"Updating ("<<shmp->ad_id<<","<<shmp->feedback<<")\n";

			double y = (double) shmp->feedback;

			ftrl.update(ad,y);
			shmp->mode = -1;
		}
	}






	
	
	return 0;
}
