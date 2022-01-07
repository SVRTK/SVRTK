/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018-2020 King's College London
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "svrtk/SphericalHarmonics.h"

using namespace std;
using namespace mirtk;

namespace svrtk {


    SphericalHarmonics::SphericalHarmonics() {

    }

    SphericalHarmonics::~SphericalHarmonics() {

    }

    Matrix SphericalHarmonics::shTransform(Matrix dirs, int lmax)
    {
        if (dirs.Cols() != 2)
            throw runtime_error("direction matrix should have 2 columns: [ azimuth elevation ]");

        Matrix SHT(dirs.Rows(), NforL (lmax));
        Vector AL(lmax+1);

        for (int i = 0; i < dirs.Rows(); i++)
        {
            double x = std::cos (dirs (i,1));
            LegendrePolynomials (AL, lmax, 0, x);
            for (int l = 0; l <= lmax; l+=2)
                SHT (i,shIndex (l,0)) = AL(l);
            for (int m = 1; m <= lmax; m++)
            {
                LegendrePolynomials (AL, lmax, m, x);
                for (int l = ( (m&1) ? m+1 : m); l <= lmax; l+=2)
                {
                    SHT(i, shIndex(l, m)) = M_SQRT2 * AL(l)*std::cos (m*dirs (i,0));
                    SHT(i, shIndex(l,-m)) = M_SQRT2 * AL(l)*std::sin (m*dirs (i,0));
                }
            }
        }
        return SHT;

    }

    void SphericalHarmonics::LegendrePolynomials(Vector& array, const int lmax, const int m, const double x)
    {
        double x2 = x*x;
        if (m && x2 >= 1.0)
        {
            for (int n = m; n <= lmax; ++n)
                array(n) = 0.0;
            return;
        }
        array(m) = 0.282094791773878;
        //m>0
        if (m) array(m) *= std::sqrt (double (2*m+1) * LegendrePolynomialsHelper (1.0-x2, 2.0*m));
        //m is odd
        if (m & 1) array(m) = -array(m);
        if (lmax == m) return;

        double f = std::sqrt (double (2*m+3));
        array(m+1) = x * f * array(m);

        for (int n = m+2; n <= lmax; n++)
        {
            array(n) = x*array(n-1) - array(n-2)/f;
            f = std::sqrt (double (4*n*n-1) / double (n*n-m*m));
            array(n) *= f;
        }
    }

    Matrix SphericalHarmonics::cartesian2spherical (Matrix xyz)
    {
        double r,x,y,z;
        Matrix az_el_r(xyz.Rows(),2);

        for (int i=0; i<xyz.Rows();i++)
        {
            x=xyz(i,0);
            y=xyz(i,1);
            z=xyz(i,2);
            r = std::sqrt (x*x + y*y + z*z);
            az_el_r(i,0) = std::atan2 (y, x);
            az_el_r(i,1) = std::acos (z/r);
        }
        return az_el_r;
    }

    Vector SphericalHarmonics::LSFit(Vector signal, Matrix dirs, int lmax)
    {
        // check that sizes correspond
        if (signal.Rows() != dirs.Rows())
            throw runtime_error("dimensions of signal and direction number do not match: " + to_string(signal.Rows()) + " " + to_string(dirs.Rows()));

        //convert to spherical coordinates
        Matrix dirs_sph;
        dirs_sph = cartesian2spherical(dirs);
        //calculate matrix of basis functions (directions x basis number)
        Matrix sht = shTransform(dirs_sph,lmax);
        //SH coefficients
        Vector c(sht.Cols());
        //calculate least square fit
        Matrix temp = sht;
        temp.Transpose();
        c=temp*signal;
        temp=temp*sht;
        temp.Invert();
        c=temp*c;
        return c;
    }

    void SphericalHarmonics::InitSHT(Matrix dirs, int lmax)
    {
        Matrix dirs_sph;
        dirs_sph = cartesian2spherical(dirs);
        //calculate matrix of basis functions (directions x basis number)
        _SHT = shTransform(dirs_sph,lmax);
        //_SHT.Print();

        //calculate inverse matrix for least square fit
        Matrix shtt = _SHT;
        shtt.Transpose();
        _iSHT = shtt*_SHT;
        _iSHT.Invert();
        _iSHT=_iSHT*shtt;


        //_iSHT.Print();
    }

    Matrix SphericalHarmonics::SHbasis(Matrix dirs, int lmax)
    {
        Matrix dirs_sph;
        dirs_sph = cartesian2spherical(dirs);
        //calculate matrix of basis functions (directions x basis number)
        Matrix SHT = shTransform(dirs_sph,lmax);
        return SHT;
    }

    void SphericalHarmonics::InitSHTRegul(Matrix dirs, double lambda, int lmax)
    {
        Matrix dirs_sph;
        dirs_sph = cartesian2spherical(dirs);
        //calculate matrix of basis functions (directions x basis number)
        _SHT = shTransform(dirs_sph,lmax);

        //calculate inverse matrix for least square fit
        Matrix shtt = _SHT;
        shtt.Transpose();
        Matrix LB = LaplaceBeltramiMatrix(lmax);
        _iSHT = shtt*_SHT+LB*lambda;
        _iSHT.Invert();
        _iSHT=_iSHT*shtt;
    }


    Matrix SphericalHarmonics::LaplaceBeltramiMatrix(int lmax)
    {
        int dim = NforL (lmax);
        int value,l,k;
        Matrix LB(dim, dim);
        //we can skip l==0, because coeff is 0 anyway
        for(l=2;l<=lmax;l=l+2)
        {
            value = l*l*(l+1)*(l+1);
            for(k=NforL(l-2); k<NforL(l); k++)
                LB(k,k)=value;
        }
        //LB.Print();
        return LB;
    }

    Vector SphericalHarmonics::Coeff2Signal(Vector c)
    {
        if (c.Rows() != _SHT.Cols())
            throw runtime_error("dimensions of SH coeffs and number of basis do not match: " + to_string(c.Rows()) + " " + to_string(_SHT.Cols()));
        return _SHT*c;
    }

    Vector SphericalHarmonics::Signal2Coeff(Vector s)
    {
        if (s.Rows() != _iSHT.Cols())
            throw runtime_error("dimensions of signal and number of directions do not match: " + to_string(s.Rows()) + " " + to_string(_iSHT.Cols()));
        return _iSHT*s;

    }

    RealImage SphericalHarmonics::Signal2Coeff(RealImage signal)
    {
        if (signal.GetT() != _iSHT.Cols())
            throw runtime_error("dimensions of signal and number of directions do not match: " + to_string(signal.GetT()) + " " + to_string(_iSHT.Cols()));

        //create image with SH coeffs
        ImageAttributes attr = signal.Attributes();
        attr._t = _iSHT.Rows();
        RealImage coeffs(attr);

        coeffs = 0;

        Vector s(signal.GetT());
        Vector a;
        int i,j,k,t;
        for(k=0;k<signal.GetZ();k++)
            for(j=0;j<signal.GetY();j++)
                for(i=0;i<signal.GetX();i++)
                {
                    if(signal(i,j,k,0)>0)
                    {
                        for(t=0;t<signal.GetT();t++)
                            s(t)=signal(i,j,k,t);

                        a=Signal2Coeff(s);

                        cout << t << endl;

                        for(t=0;t<coeffs.GetT();t++)
                            coeffs(i,j,k,t)=a(t);
                    }
                }

        return coeffs;
    }


    RealImage SphericalHarmonics::Coeff2Signal(RealImage coeffs)
    {
        if (coeffs.GetT() != _SHT.Cols())
            throw runtime_error("dimensions of SH coeffs and number of basis do not match: " + to_string(coeffs.GetT()) + " " + to_string(_SHT.Cols()));

        //create image with SH coeffs
        ImageAttributes attr = coeffs.Attributes();
        attr._t = _SHT.Rows();
        RealImage signal(attr);
        Vector c(coeffs.GetT());
        Vector s;
        int i,j,k,t;
        for(k=0;k<coeffs.GetZ();k++)
            for(j=0;j<coeffs.GetY();j++)
                for(i=0;i<coeffs.GetX();i++)
                {
                    //it should be ok - the first coeff should not be negative for positive signal
                    //if(coeffs(i,j,k,0)>0)
                    //{
                    for(t=0;t<coeffs.GetT();t++)
                        c(t)=coeffs(i,j,k,t);
                    s=Coeff2Signal(c);
                    for(t=0;t<signal.GetT();t++)
                        signal(i,j,k,t)=s(t);
                    //}
                }


        return signal;
    }



} // namespace svrtk
