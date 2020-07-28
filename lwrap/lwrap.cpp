#include <libint2.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace py = pybind11;


struct engine{
    private:
        libint2::Engine _integrator;
        libint2::Operator _opt;
		//int _nuclear = 0;
        //std::vector<std::vector<float>> _retbuff; //return buffer
    public:
        engine(){libint2::initialize();
                 _opt = libint2::Operator::coulomb;};
        ~engine(){libint2::finalize();};
        void setup(){};
        void set_operator_coulomb(){_opt = libint2::Operator::coulomb;}
        void set_operator_erf(){_opt = libint2::Operator::erf_coulomb;}
        void set_operator_erfc(){_opt = libint2::Operator::erfc_coulomb;}
        void set_operator_overlap(){_opt = libint2::Operator::overlap;}
        void set_operator_nuclear(){_opt = libint2::Operator::nuclear;}
        void set_operator_emultipole(){_opt = libint2::Operator::emultipole3;}
        void set_operator_kinetic(){_opt = libint2::Operator::kinetic;}
        void set_integrator_params(double w){_integrator.set_params(w);}
		
        void set_braket_xsxs(){_integrator.set(libint2::BraKet::xs_xs);}
        std::vector<double> get(){return std::vector<double>(4,2.0);};

		void set_charges(std::string geometry_p, std::string geometry_q){
			//_nuclear = 1;
			
			ifstream input_file_p(geometry_p);
			vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
			//libint2::BasisSet obs_p(basis_p, atoms_p);

			ifstream input_file_q(geometry_q);
			vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
			//libint2::BasisSet obs_q(basis_q, atoms_q);

			/*

			std::vector<std::pair<double,std::array<double,3>>> q;
			for(const auto& atom : atoms_q) {
				q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x*1.8897261339212517, atom.y*1.8897261339212517, atom.z*1.8897261339212517}}} );
			} */

			std::vector<std::pair<double, std::array<double, 3>>> q;
			q.reserve(atoms_q.size());
			for (const auto& atom : atoms_q) {
			//cout << static_cast<double>(atom.atomic_number) << endl;
			q.emplace_back(static_cast<double>(atom.atomic_number),
							std::array<double, 3>{{atom.x, atom.y, atom.z}});
							//std::array<double, 3>{{static_cast<double>(atom.atomic_number), static_cast<double>(atom.atomic_number), static_cast<double>(atom.atomic_number)}});
			}
			

			_integrator.set_params(q);
			//_integrator.set_q(q);
			

			//_integrator.set_params(libint2::make_point_charges(atoms_q));
			} 


			
			//_integrator.set_params(libint2::make_point_charges(atoms_q));}

        
        //  Two body interaction integrals
        void setup_pqrs(std::string geometry_p,
              std::string basis_p,
              std::string geometry_q,
              std::string basis_q,
              std::string geometry_r,
              std::string basis_r,
              std::string geometry_s,
              std::string basis_s,
              int angmom_add){

	   //libint2::Operator opt = libint2::Operator::coulomb;
	   
           // read input files
	   ifstream input_file_p(geometry_p);
	   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
	   libint2::BasisSet obs_p(basis_p, atoms_p);

	   ifstream input_file_q(geometry_q);
	   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
	   libint2::BasisSet obs_q(basis_q, atoms_q);

	   ifstream input_file_r(geometry_r);
	   vector<libint2::Atom> atoms_r = libint2::read_dotxyz(input_file_r);
	   libint2::BasisSet obs_r(basis_r, atoms_r);

	   ifstream input_file_s(geometry_s);
	   vector<libint2::Atom> atoms_s = libint2::read_dotxyz(input_file_s);
	   libint2::BasisSet obs_s(basis_s, atoms_s);

	   //determine max angular momentum in basis, and max number of primitives
	   std::size_t maxl = obs_q.max_l();
	   if(obs_p.max_l()>obs_q.max_l()){
	       maxl = obs_p.max_l();
	       if(obs_r.max_l()>obs_p.max_l()){
		  maxl = obs_r.max_l();
	       }
	   }
           maxl = maxl + angmom_add; 

	   std::size_t maxnprim = obs_q.max_nprim();
	   if(obs_p.max_nprim()>obs_q.max_nprim()){
	       maxnprim = obs_p.max_nprim();
	       if(obs_r.max_nprim()>obs_p.max_nprim()){
		  maxnprim = obs_r.max_nprim();
	       }
	   }


	   // initialize the integral engine
	   _integrator = libint2::Engine(_opt,
			maxnprim,    // max # of primitives in shells this engine will accept
			maxl         // max angular momentum of shells this engine will accept
		       );
	   //_integrator.set(libint2::BraKet::xs_xx);
        
        };


        void setup_pqr(std::string geometry_p,
              std::string basis_p,
              std::string geometry_q,
              std::string basis_q,
              std::string geometry_r,
              std::string basis_r,
              int angmom_add){

	   //libint2::Operator opt = libint2::Operator::coulomb;
	   
           // read input files
	   ifstream input_file_p(geometry_p);
	   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
	   libint2::BasisSet obs_p(basis_p, atoms_p);

	   ifstream input_file_q(geometry_q);
	   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
	   libint2::BasisSet obs_q(basis_q, atoms_q);

	   ifstream input_file_r(geometry_r);
	   vector<libint2::Atom> atoms_r = libint2::read_dotxyz(input_file_r);
	   libint2::BasisSet obs_r(basis_r, atoms_r);

	   //determine max angular momentum in basis, and max number of primitives
	   std::size_t maxl = obs_q.max_l();
	   if(obs_p.max_l()>obs_q.max_l()){
	       maxl = obs_p.max_l();
	       if(obs_r.max_l()>obs_p.max_l()){
		  maxl = obs_r.max_l();
	       }
	   }
           maxl = maxl + angmom_add; 

	   std::size_t maxnprim = obs_q.max_nprim();
	   if(obs_p.max_nprim()>obs_q.max_nprim()){
	       maxnprim = obs_p.max_nprim();
	       if(obs_r.max_nprim()>obs_p.max_nprim()){
		  maxnprim = obs_r.max_nprim();
	       }
	   }


	   // initialize the integral engine
	   _integrator = libint2::Engine(_opt,
			maxnprim,    // max # of primitives in shells this engine will accept
			maxl         // max angular momentum of shells this engine will accept
		       );
	   _integrator.set(libint2::BraKet::xs_xx);
        
        };


        std::vector<vector<vector<double>>> get_pqr(std::string geometry_p,
              std::string basis_p,
              std::string geometry_q,
              std::string basis_q,
              std::string geometry_r,
              std::string basis_r){

	   // Compute ( p q | V | p q )
	   // where V is the coulomb operator
           

           // read input files
           ifstream input_file_p(geometry_p);
           vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
           libint2::BasisSet obs_p(basis_p, atoms_p);
           
           ifstream input_file_q(geometry_q);
           vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
           libint2::BasisSet obs_q(basis_q, atoms_q);
           
           ifstream input_file_r(geometry_r);
           vector<libint2::Atom> atoms_r = libint2::read_dotxyz(input_file_r);
           libint2::BasisSet obs_r(basis_r, atoms_r);

           //std::vector<float> results; // list for results, to be returned to python

	   // calculate and gather results
	   auto shell2bf_p = obs_p.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
	   auto shell2bf_q = obs_q.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
					// ...

	   auto shell2bf_r = obs_r.shell2bf(); // maps shell index to basis function index

	   const auto& buf_vec = _integrator.results(); // will point to computed shell sets
						  // const auto& is very important!

           int nq(obs_q.nbf());
           int np(obs_p.nbf());
           int nr(obs_r.nbf());
           std::vector<std::vector<std::vector<double>>> results(np, std::vector<std::vector<double>>(nq,std::vector<double>(nr, 0.0)));
           //vector<vector<vector<double>>> f(3, vector<vector<double>>(4, vector<double>(5))); 
		for(auto s1=0; s1!=obs_p.size(); ++s1) {
		  for(auto s2=0; s2!=obs_q.size(); ++s2) {
		     for(auto s3=0; s3!=obs_r.size(); ++s3) {
		       //for all shells in osb
		       //cout << "compute shell set {" << s1 << "," << s2 << "} ... ";
		       _integrator.compute(obs_p[s1], obs_q[s2], obs_r[s3]);

		       //cout << "done" << endl;
		       auto ints_shellset = buf_vec[0];  // location of the computed integrals
		       if (ints_shellset == nullptr)
			 continue;  // nullptr returned if the entire shell-set was screened out

		       auto bf1 = shell2bf_p[s1];  // first basis function in first shell
		       auto n1 = obs_p[s1].size(); // number of basis functions in first shell
		       auto bf2 = shell2bf_q[s2];  // first basis function in second shell
		       auto n2 = obs_q[s2].size(); // number of basis functions in second shell
		       auto bf3 = shell2bf_r[s3];  // first basis function in second shell
		       auto n3 = obs_r[s3].size(); // number of basis functions in second shell
		       //for(auto n = 0; n<ints_shellset.size(); ++n){
		       //   cout << ints_shellset[n] << endl;
		       //}
		       // integrals are packed into ints_shellset in row-major (C) form
		       // this iterates over integrals in this order
		       for(int f1=0; f1!=n1; ++f1){
			 for(int f2=0; f2!=n2; ++f2){
			   for(int f3=0; f3!=n3; ++f3){
			     //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
			     //results.push_back(ints_shellset[f1*n2*n3 +f2*n3 + f3]);
                             results[bf1+f1][bf2+f2][bf3+f3] = ints_shellset[f1*n2*n3 + f2*n3 + f3];
			   }
			 }
		      }
		   }
		   }
		   }




	   return results;};





        // Two index integrals
 
        void setup_pq(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q){

		   //libint2::Operator opt = libint2::Operator::coulomb;

		   // read input files
		   ifstream input_file_p(geometry_p);
		   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
		   libint2::BasisSet obs_p(basis_p, atoms_p);

		   ifstream input_file_q(geometry_q);
		   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
		   libint2::BasisSet obs_q(basis_q, atoms_q);


		   //determine max angular momentum in basis, and max number of primitives
		   std::size_t maxl = obs_q.max_l();
		   if(obs_p.max_l()>obs_q.max_l()){
		       maxl = obs_p.max_l();
		   }

		   std::size_t maxnprim = obs_q.max_nprim();
		   if(obs_p.max_nprim()>obs_q.max_nprim()){
		       maxnprim = obs_p.max_nprim();
		   }


		   // initialize the integral engine
		   _integrator = libint2::Engine(_opt,
				maxnprim,    // max # of primitives in shells this engine will accept
				maxl         // max angular momentum of shells this engine will accept
			       );
		   //_integrator.set(libint2::BraKet::xs_xs);
                   //int nq(obs_q.nbf());
                   //int np(obs_p.nbf());
                   //_retbuff = std::vector<std::vector<float>>  (np, vector<float>(nq,0.0));
        
};




        std::vector<vector<double>> get_pq(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q){

	   // Compute ( p | V | q )
	   // where V is the coulomb operator

 // read input files
                   ifstream input_file_p(geometry_p);
                   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
                   libint2::BasisSet obs_p(basis_p, atoms_p);

                   ifstream input_file_q(geometry_q);
                   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
                   libint2::BasisSet obs_q(basis_q, atoms_q);

				   /*if (_nuclear == 1){
					   std::vector<std::pair<double,std::array<double,3>>> q;
						for(const auto& atom : atoms_q) {
							q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
						}

						_integrator.set_params(q);
					   

				   //}*/

           //std::vector<float> results; // list for results, to be returned to python

	   // calculate and gather results
	   auto shell2bf_p = obs_p.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
	   auto shell2bf_q = obs_q.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
					// ...
           int nq(obs_q.nbf());
           int np(obs_p.nbf());
           std::vector<std::vector<double>> results(np, vector<double>(nq,0.0));
 

	   const auto& buf_vec = _integrator.results(); // will point to computed shell sets
						  // const auto& is very important!

	   for(auto s1=0; s1!=obs_p.size(); ++s1) {
	     for(auto s2=0; s2!=obs_q.size(); ++s2) {
	       //for all shells in osb
	       _integrator.compute(obs_p[s1], obs_q[s2]);
	       auto ints_shellset = buf_vec[0];  // location of the computed integrals
	       if (ints_shellset == nullptr)
		   continue;  // nullptr returned if the entire shell-set was screened out

	       auto bf1 = shell2bf_p[s1];  // first basis function in first shell
	       auto n1 = obs_p[s1].size(); // number of basis functions in first shell
	       auto bf2 = shell2bf_q[s2];  // first basis function in second shell
	       auto n2 = obs_q[s2].size(); // number of basis functions in second shell
        
               // integrals are packed into ints_shellset in row-major (C) form
	       // this iterates over integrals in this order
	       for(int f1=0; f1!=n1; ++f1){
		 for(int f2=0; f2!=n2; ++f2){
		   //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
		   //results.push_back(ints_shellset[f1*n2+f2]);
                   results[bf1+f1][bf2+f2] = ints_shellset[f1*n2+f2];
		 }
	      }
	   }
	   }

	   return results;};

std::vector<vector<vector<double>>> get_pq_multipole(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q){

	   // Compute ( p | V | q )
	   // where V is the coulomb operator

 // read input files
                   ifstream input_file_p(geometry_p);
                   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
                   libint2::BasisSet obs_p(basis_p, atoms_p);

                   ifstream input_file_q(geometry_q);
                   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
                   libint2::BasisSet obs_q(basis_q, atoms_q);

           //std::vector<float> results; // list for results, to be returned to python

	   // calculate and gather results
	   auto shell2bf_p = obs_p.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
	   auto shell2bf_q = obs_q.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
					// ...
        int nq(obs_q.nbf());
        int np(obs_p.nbf());
        
		std::vector<std::vector<std::vector<double>>> results(np, std::vector<std::vector<double>>(nq,std::vector<double>(20, 0.0)));
	    
		const auto& buf_vec = _integrator.results(); // will point to computed shell sets
						  // const auto& is very important!

	    for(auto s1=0; s1!=obs_p.size(); ++s1) {
	     for(auto s2=0; s2!=obs_q.size(); ++s2) {
	       //for all shells in osb
	       _integrator.compute(obs_p[s1], obs_q[s2]);
	       auto ints_shellset = buf_vec[0];  // location of the computed integrals
	       if (ints_shellset == nullptr)
		   continue;  // nullptr returned if the entire shell-set was screened out

	       auto bf1 = shell2bf_p[s1];  // first basis function in first shell
	       auto n1 = obs_p[s1].size(); // number of basis functions in first shell
	       auto bf2 = shell2bf_q[s2];  // first basis function in second shell
	       auto n2 = obs_q[s2].size(); // number of basis functions in second shell
        
               // integrals are packed into ints_shellset in row-major (C) form
	       // this iterates over integrals in this order
	        for(int f1=0; f1!=n1; ++f1){
		      for(int f2=0; f2!=n2; ++f2){
		   //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
		   //results.push_back(ints_shellset[f1*n2+f2]);
                for(int j = 0; j!=20; ++j){
                   results[bf1+f1][bf2+f2][j] = ints_shellset[f1*n2+f2 + j*n2*n1];
				   }
		      }
	        }
	   }
	   }

	   return results;};


        // Cauchy-Schwartz screening integrals
 
        void setup_pqpq(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q){

		   //libint2::Operator opt = libint2::Operator::coulomb;

		   // read input files
		   ifstream input_file_p(geometry_p);
		   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
		   libint2::BasisSet obs_p(basis_p, atoms_p);

		   ifstream input_file_q(geometry_q);
		   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
		   libint2::BasisSet obs_q(basis_q, atoms_q);


		   //determine max angular momentum in basis, and max number of primitives
		   std::size_t maxl = obs_q.max_l();
		   if(obs_p.max_l()>obs_q.max_l()){
		       maxl = obs_p.max_l();
		   }

		   std::size_t maxnprim = obs_q.max_nprim();
		   if(obs_p.max_nprim()>obs_q.max_nprim()){
		       maxnprim = obs_p.max_nprim();
		   }


		   // initialize the integral engine
		   _integrator = libint2::Engine(_opt,
				maxnprim,    // max # of primitives in shells this engine will accept
				maxl         // max angular momentum of shells this engine will accept
			       );
	};


std::vector<std::vector<double>> get_pqpq(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q){

	   // Compute ( p q | V | p q )
	   // where V is the coulomb operator

 // read input files
                   ifstream input_file_p(geometry_p);
                   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
                   libint2::BasisSet obs_p(basis_p, atoms_p);

                   ifstream input_file_q(geometry_q);
                   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
                   libint2::BasisSet obs_q(basis_q, atoms_q);

           //std::vector<float> results; // list for results, to be returned to python

	   // calculate and gather results
	   auto shell2bf_p = obs_p.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1	   
           auto shell2bf_q = obs_q.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
					// ...
	   const auto& buf_vec = _integrator.results(); // will point to computed shell sets
						  // const auto& is very important!
           int nq(obs_q.nbf());
           int np(obs_p.nbf());
           std::vector<std::vector<double>> results(np, vector<double>(nq,0.0));

	   for(auto s1=0; s1!=obs_p.size(); ++s1) {
	     for(auto s2=0; s2!=obs_q.size(); ++s2) {
	       //for all shells in osb
	       _integrator.compute(obs_p[s1], obs_q[s2], obs_p[s1], obs_q[s2]);
	       auto ints_shellset = buf_vec[0];  // location of the computed integrals
	       if (ints_shellset == nullptr)
		   continue;  // nullptr returned if the entire shell-set was screened out

	       auto bf1 = shell2bf_p[s1];  // first basis function in first shell
	       auto n1 = obs_p[s1].size(); // number of basis functions in first shell
	       auto bf2 = shell2bf_q[s2];  // first basis function in second shell
	       auto n2 = obs_q[s2].size(); // number of basis functions in second shell
        
               // integrals are packed into ints_shellset in row-major (C) form
	       // this iterates over integrals in this order
	       for(int f1=0; f1!=n1; ++f1){
		 for(int f2=0; f2!=n2; ++f2){
		   //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
		   //results.push_back(ints_shellset[f1*n2+f2]);
                   results[bf1+f1][bf2+f2] = ints_shellset[f1*n2+f2];
		 }
	      }
	   }
	   }

	   return results;};



        std::vector<std::vector<std::vector<std::vector<double>>>> get_pqrs(std::string geometry_p,
                                    std::string basis_p,
                                    std::string geometry_q,
                                    std::string basis_q,
                                    std::string geometry_r,
                                    std::string basis_r,
                                    std::string geometry_s,
                                    std::string basis_s){

	   // Compute ( p q | V | p q )
	   // where V is the coulomb operator

 // read input files
                   ifstream input_file_p(geometry_p);
                   vector<libint2::Atom> atoms_p = libint2::read_dotxyz(input_file_p);
                   libint2::BasisSet obs_p(basis_p, atoms_p);

                   ifstream input_file_q(geometry_q);
                   vector<libint2::Atom> atoms_q = libint2::read_dotxyz(input_file_q);
                   libint2::BasisSet obs_q(basis_q, atoms_q);

                   ifstream input_file_r(geometry_r);
                   vector<libint2::Atom> atoms_r = libint2::read_dotxyz(input_file_r);
                   libint2::BasisSet obs_r(basis_r, atoms_r);

                   ifstream input_file_s(geometry_s);
                   vector<libint2::Atom> atoms_s = libint2::read_dotxyz(input_file_s);
                   libint2::BasisSet obs_s(basis_s, atoms_s);


           //std::vector<float> results; // list for results, to be returned to python

	   // calculate and gather results
	   auto shell2bf_p = obs_p.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1	   
           auto shell2bf_q = obs_q.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
           auto shell2bf_r = obs_r.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1	   
           auto shell2bf_s = obs_s.shell2bf(); // maps shell index to basis function index
					// shell2bf[0] = index of the first basis function in shell 0
					// shell2bf[1] = index of the first basis function in shell 1
							// ...
	   const auto& buf_vec = _integrator.results(); // will point to computed shell sets
						  // const auto& is very important!
           int nq(obs_q.nbf());
           int np(obs_p.nbf());
           int nr(obs_r.nbf());
           int ns(obs_s.nbf());

           //std::vector<std::vector<double>> results(np, vector<double>(nq,0.0));

           std::vector<std::vector<std::vector<std::vector<double>>>> results(np, 
                                                                              std::vector<std::vector<std::vector<double>>>(nq,
                                                                              std::vector<std::vector<double>>(nr, 
                                                                              std::vector<double>(ns, 0.0))));
           
	   for(auto s1=0; s1!=obs_p.size(); ++s1) {
	     for(auto s2=0; s2!=obs_q.size(); ++s2) {
  	       for(auto s3=0; s3!=obs_r.size(); ++s3) {
	         for(auto s4=0; s4!=obs_s.size(); ++s4) {
  	           //for all shells in osb
	           _integrator.compute(obs_p[s1], obs_q[s2], obs_r[s3], obs_s[s4]);
	           auto ints_shellset = buf_vec[0];  // location of the computed integrals
	           if (ints_shellset == nullptr)
	             continue;  // nullptr returned if the entire shell-set was screened out
    
	           auto bf1 = shell2bf_p[s1];  // first basis function in first shell
	           auto n1 = obs_p[s1].size(); // number of basis functions in first shell
	           auto bf2 = shell2bf_q[s2];  // first basis function in second shell
	           auto n2 = obs_q[s2].size(); // number of basis functions in second shell
     
	           auto bf3 = shell2bf_r[s3];  // first basis function in first shell
	           auto n3 = obs_r[s3].size(); // number of basis functions in first shell
	           auto bf4 = shell2bf_s[s4];  // first basis function in second shell
	           auto n4 = obs_s[s4].size(); // number of basis functions in second shell
        
                   // integrals are packed into ints_shellset in row-major (C) form
	           // this iterates over integrals in this order
	           for(int f1=0; f1!=n1; ++f1){
		     for(int f2=0; f2!=n2; ++f2){
                       for(int f3=0; f3!=n3; ++f3){
                         for(int f4=0; f4!=n4; ++f4){
		           //cout << "  " << bf1+f1 << " " << bf2+f2 << " " << ints_shellset[f1*n2+f2] << endl;
		           //results.push_back(ints_shellset[f1*n2+f2]);
                           results[bf1+f1][bf2+f2][bf3 + f3][bf4 + f4] = ints_shellset[f1*n2*n3*n4+f2*n3*n4 + f3*n4 + f4];
		         }
                       }
                     }
	           }
	         }
	       }
             }
           }

	   return results;};



};


PYBIND11_MODULE(lwrap, m) {

    py::class_<engine>(m, "engine")
        .def(py::init())
        .def("setup", &engine::setup)
        .def("get", &engine::get)
        .def("set_operator_overlap", &engine::set_operator_overlap)
        .def("set_operator_kinetic", &engine::set_operator_kinetic)
        .def("set_operator_nuclear", &engine::set_operator_nuclear)
        .def("set_operator_coulomb", &engine::set_operator_coulomb)
		.def("set_operator_emultipole", &engine::set_operator_emultipole)
        .def("set_braket_xsxs", &engine::set_braket_xsxs)
        .def("set_integrator_params", &engine::set_integrator_params)
		.def("set_charges", &engine::set_charges)
        .def("set_operator_erf", &engine::set_operator_erf)
        .def("set_operator_erfc", &engine::set_operator_erfc)
        .def("setup_pqpq", &engine::setup_pqpq)
        .def("get_pqpq", &engine::get_pqpq)
        .def("setup_pqrs", &engine::setup_pqrs)
        .def("get_pqrs", &engine::get_pqrs)
        .def("setup_pqr", &engine::setup_pqr)
        .def("get_pqr", &engine::get_pqr)
        .def("setup_pq", &engine::setup_pq)
        .def("get_pq", &engine::get_pq)
		.def("get_pq_multipole", &engine::get_pq_multipole);
}
