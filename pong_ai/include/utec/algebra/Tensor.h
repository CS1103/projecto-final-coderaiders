#pragma once
#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace utec::algebra {
    template<typename T , size_t R> class Tensor;
    template<typename T , size_t R>
    Tensor<T,R> transpose_2d(const Tensor<T,R>&);
    template<typename T , size_t R>
    Tensor<T,R> matrix_product(const Tensor<T,R>&, const Tensor<T,R>&);
    template<typename T , size_t R>
    class Tensor {
    public:
        using dim_array = std::array<size_t,R>;
        using pos_array = std::array<size_t,R>;
    private:
        dim_array dims_{};
        std::vector<T> data_{};
        static size_t product(const dim_array& d){
            return std::accumulate(d.begin(), d.end(), size_t{1},
                                   std::multiplies<size_t>{});
        }
        static dim_array make_stride(const dim_array& d){
            dim_array s{};
            s[R-1]=1;
            for(int i=int(R)-2;i>=0;--i) s[i]=s[i+1]*d[i+1];
            return s;
        }
        static size_t coord2index(const pos_array& p,const dim_array& d){
            auto s=make_stride(d); size_t idx=0;
            for(size_t i=0;i<R;++i) idx+=p[i]*s[i];
            return idx;
        }
        static pos_array index2coord(size_t idx,const dim_array& d){
            auto s=make_stride(d); pos_array c{};
            for(size_t i=0;i<R;++i){ c[i]=idx/s[i]; idx%=s[i]; }
            return c;
        }
        template<typename OP>
        Tensor elementwise(const Tensor& b,OP op) const{
            if(!broadcastable(dims_,b.dims_))
                throw std::invalid_argument("incompatible shapes for broadcasting");
            dim_array rd{};
            for(size_t i=0;i<R;++i) rd[i]=std::max(dims_[i],b.dims_[i]);
            Tensor r(rd);
            for(size_t k=0;k<r.size();++k){
                auto coord=index2coord(k,rd);
                pos_array ca{},cb{};
                for(size_t i=0;i<R;++i){
                    ca[i]=(dims_[i]==1?0:coord[i]);
                    cb[i]=(b.dims_[i]==1?0:coord[i]);
                }
                r.data_[k]=op(at_raw(ca),b.at_raw(cb));
            }
            return r;
        }
        static bool broadcastable(const dim_array& a,const dim_array& b){
            for(size_t i=0;i<R;++i)
                if(a[i]!=b[i] && a[i]!=1 && b[i]!=1) return false;
            return true;
        }
    public:
        Tensor()=default;
        template<typename... Dims,
                typename = std::enable_if_t<(sizeof...(Dims)==R)>>
        explicit Tensor(Dims... ds){
            size_t tmp[]{ static_cast<size_t>(ds)... };
            for(size_t i=0;i<R;++i) dims_[i]=tmp[i];
            data_.resize(product(dims_));
        }
        explicit Tensor(const dim_array& d){ dims_=d; data_.resize(product(d)); }
        explicit Tensor(std::initializer_list<size_t> dims) {
            for (size_t i = 0; i < dims.size(); ++i)
                dims_[i] = *(dims.begin() + i);
            data_.resize(product(dims_));
        }
        dim_array shape() const            { return dims_; }
        size_t    size()  const            { return data_.size(); }
        size_t rows() const {
            static_assert(R == 2, "rows() solo válido para tensores 2D");
            return dims_[0];
        }

        size_t cols() const {
            static_assert(R == 2, "cols() solo válido para tensores 2D");
            return dims_[1];
        }
        void reshape(const dim_array& new_shape) {
            if (product(new_shape) != product(dims_)) {
                throw std::invalid_argument("reshape: number of elements must not change");
            }
            dims_ = new_shape;
        }
        template<typename... Idx,
                typename = std::enable_if_t<(sizeof...(Idx)==R)>>
        T& operator()(Idx... idx){
            size_t tmp[]{ static_cast<size_t>(idx)... };
            pos_array p{}; for(size_t i=0;i<R;++i) p[i]=tmp[i];
            return data_[coord2index(p,dims_)];
        }
        template<typename... Idx,
                typename = std::enable_if_t<(sizeof...(Idx)==R)>>
        const T& operator()(Idx... idx) const{
            size_t tmp[]{ static_cast<size_t>(idx)... };
            pos_array p{}; for(size_t i=0;i<R;++i) p[i]=tmp[i];
            return data_[coord2index(p,dims_)];
        }
        T& operator[](size_t i)             { return data_[i]; }
        const T& operator[](size_t i) const { return data_[i]; }
        auto begin()       { return data_.begin(); }
        auto end()         { return data_.end();   }
        auto cbegin() const{ return data_.cbegin();}
        auto cend()   const{ return data_.cend();  }
        void fill(const T& v){ std::fill(begin(),end(),v); }
        Tensor& operator=(std::initializer_list<T> lst){
            if(lst.size()!=data_.size())
                throw std::invalid_argument("initializer size != tensor size");
            std::copy(lst.begin(),lst.end(),begin()); return *this;
        }
        Tensor<T, 2> transpose_2d() const {
            static_assert(R == 2, "transpose_2d solo es válido para tensores 2D");
            Tensor<T, 2> result({dims_[1], dims_[0]});
            for (size_t i = 0; i < dims_[0]; ++i)
                for (size_t j = 0; j < dims_[1]; ++j)
                    result(j, i) = (*this)(i, j);
            return result;
        }
        Tensor operator+(const Tensor& b) const{ return elementwise(b,[](T a,T b){return a+b;}); }
        Tensor operator-(const Tensor& b) const{ return elementwise(b,[](T a,T b){return a-b;}); }
        Tensor operator*(const Tensor& b) const{ return elementwise(b,[](T a,T b){return a*b;}); }
        Tensor operator+(T s) const{ Tensor r=*this; for(auto&v:r)v+=s; return r; }
        Tensor operator-(T s) const{ Tensor r=*this; for(auto&v:r)v-=s; return r; }
        Tensor operator*(T s) const{ Tensor r=*this; for(auto&v:r)v*=s; return r; }
        Tensor operator/(T s) const{ Tensor r=*this; for(auto&v:r)v/=s; return r; }
        friend Tensor operator+(T s,const Tensor& t){ return t+s; }
        friend std::ostream& operator<<(std::ostream& os,const Tensor& t){
            if constexpr(R==1){
                for(size_t i=0;i<t.size();++i){ os<<t.data_[i]; if(i+1<t.size()) os<<' '; }
                return os;
            }
            t.print_rec(os,0,0); return os;
        }
    private:
        void print_rec(std::ostream& os,size_t depth,size_t offset) const{
            if(depth==0) os<<"{\n";
            if(depth==R-1){
                for(size_t i=0;i<dims_[depth];++i){
                    os<<data_[offset+i];
                    if(i+1<dims_[depth]) os<<' ';
                }
                os<<'\n';
            }else{
                size_t stride=1; for(size_t k=depth+1;k<R;++k) stride*=dims_[k];
                for(size_t i=0;i<dims_[depth];++i){
                    if(depth<R-2) os<<"{\n";
                    print_rec(os,depth+1,offset+i*stride);
                    if(depth<R-2) os<<"}\n";
                }
            }
            if(depth==0) os<<"}";
        }
        T& at_raw(const pos_array& p){ return data_[coord2index(p,dims_)]; }
        const T& at_raw(const pos_array& p) const{ return data_[coord2index(p,dims_)]; }
        friend Tensor matrix_product<T,R>(const Tensor&,const Tensor&);
    };
    template<typename T,size_t R>
    Tensor<T,R> transpose_2d(const Tensor<T,R>& A){
        static_assert(R>=2,"rank must be >=2");
        auto nd=A.dims_; std::swap(nd[R-2],nd[R-1]);
        Tensor<T,R> Rm(nd);
        for(size_t i=0;i<A.size();++i){
            auto c=Tensor<T,R>::index2coord(i,nd);
            typename Tensor<T,R>::pos_array src=c;
            std::swap(src[R-2],src[R-1]);
            Rm[i]=A.at_raw(src);
        }
        return Rm;
    }
    template<typename T,size_t R>
    Tensor<T,R> matrix_product(const Tensor<T,R>& A,const Tensor<T,R>& B){
        static_assert(R>=2,"rank must be >=2");
        const auto ad=A.dims_, bd=B.dims_;
        if(ad[R-1]!=bd[R-2]) throw std::invalid_argument("matmul incompatible dims");
        typename Tensor<T,R>::dim_array rd=ad; rd[R-1]=bd[R-1];
        Tensor<T,R> C(rd);
        size_t m=ad[R-2], k=ad[R-1], n=bd[R-1];
        size_t batch = C.size()/(m*n);

        for(size_t b=0;b<batch;++b){
            size_t offsetA=b*m*k, offsetB=b*k*n, offsetC=b*m*n;
            for(size_t i=0;i<m;++i)
                for(size_t j=0;j<n;++j){
                    T acc{};
                    for(size_t t=0;t<k;++t) acc+=A[offsetA+i*k+t]*B[offsetB+t*n+j];
                    C[offsetC+i*n+j]=acc;
                }
        }
        return C;
    }

}