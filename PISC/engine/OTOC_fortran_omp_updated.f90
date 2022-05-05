module otoc_tools
	use omp_lib
	implicit none
	private
	public :: pos_matrix_elts
	public :: compute_pos_mat_arr
	public :: linop_matrix_elts
	public :: quadop_matrix_elts
	public :: quadop_matrix_arr
	public :: quadop_matrix_arr_t
	public :: corr_mc_elts
	public :: corr_mc_arr
	public :: corr_mc_arr_t
	public :: therm_corr_elts
	public :: kubo_corr_elts
	public :: therm_corr_arr_t
	contains
		subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
			integer, intent(in) :: n,k , len1vecs,len2vecs,lenx
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			!f2py real(kind=8), intent(in,out,copy) :: pos_mat_elt
			integer :: i,j
			real(kind=8), intent(inout) :: pos_mat_elt
			pos_mat_elt = 0.0d0
			do i = 1,len1vecs
					pos_mat_elt = pos_mat_elt + vecs(i,n)*vecs(i,k)*x_arr(i)*dx*dy !! Change here for 1D 
			end do
		end subroutine pos_matrix_elts

		subroutine compute_pos_mat_arr(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr,lenk,key,pos_mat)
			integer, intent(in) ::  len1vecs,len2vecs,lenx,lenk,n
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer :: i,j
			character*3, intent(in) :: key
			!f2py real(kind=8), dimension(lenk), intent(in,out,copy) :: pos_mat
			real(kind=8), dimension(lenk), intent(inout) :: pos_mat
			!$OMP PARALLEL DO PRIVATE(i)
			do i=1,lenk
				if (key=='ket') then
					call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr(i),pos_mat(i))
				else
					call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr(i),n,pos_mat(i))
				end if
			end do
			!$OMP END PARALLEL DO
		end subroutine compute_pos_mat_arr

		subroutine linop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,vals_arr,lenv,n,m,t,key,linop_mat_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: n, m
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t, mass
			integer :: i
			character*1, intent(in) :: key
			!f2py complex, intent(in,out,copy) :: linop_mat_elt
			complex, intent(inout) :: linop_mat_elt
			real(kind=8) :: E_nm, pos_mat_elt

			call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,m,pos_mat_elt)
			E_nm = vals_arr(n) - vals_arr(m)
			if (key=='q') then
				linop_mat_elt = exp(cmplx(0.0,1.0)*E_nm*t)*pos_mat_elt
			else if (key=='p') then
				linop_mat_elt = cmplx(0.0,1.0)*mass*exp(cmplx(0.0,1.0)*E_nm*t)*pos_mat_elt
			end if
		end subroutine linop_matrix_elts
				
		subroutine quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t1,t2,key,quadop_mat_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: lenk, n, m
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t1,t2,mass
			integer :: i
			character*2, intent(in) :: key
			!f2py complex, intent(in,out,copy) :: quadop_mat_elt
			complex, intent(inout) :: quadop_mat_elt
			real(kind=8) :: E_km, E_nk
			real(kind=8), dimension(lenk) :: x_nk, x_km
	
			quadop_mat_elt = cmplx(0.0,0.0)
			x_nk = 0.0d0
			x_km = 0.0d0
			if(n==m) then
				call compute_pos_mat_arr(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr,lenk,'ket',x_nk)
				do i=1,lenk
					E_nk = vals_arr(n) - vals_arr(k_arr(i))
					if (key=='qq') then
						quadop_mat_elt = quadop_mat_elt + exp(cmplx(0.0,1.0)*E_nk*(t1-t2))*x_nk(i)*x_nk(i)
					else if(key=='qp') then
						quadop_mat_elt = quadop_mat_elt - cmplx(0.0,1.0)*mass*exp(cmplx(0.0,1.0)*E_nk*(t1-t2))*x_nk(i)*x_nk(i)*E_nk
					else if(key=='pq') then
						quadop_mat_elt = quadop_mat_elt + cmplx(0.0,1.0)*mass*exp(cmplx(0.0,1.0)*E_nk*(t1-t2))*x_nk(i)*x_nk(i)*E_nk
					else if(key=='pp') then
						quadop_mat_elt = quadop_mat_elt + mass*mass*exp(cmplx(0.0,1.0)*E_nk*(t1-t2))*x_nk(i)*x_nk(i)*E_nk*E_nk
					else if(key=='cm') then
						quadop_mat_elt = quadop_mat_elt - 2*mass*(x_nk(i)*x_nk(i)*E_nk*cos(E_nk*(t1-t2)))
					end if
				end do

			else
				call compute_pos_mat_arr(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr,lenk,'ket',x_nk)
				call compute_pos_mat_arr(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,m,k_arr,lenk,'bra',x_km)
				do i=1,lenk
					E_nk = vals_arr(n) - vals_arr(k_arr(i))
					E_km = vals_arr(k_arr(i)) - vals_arr(m)
					if (key=='qq') then
						quadop_mat_elt = quadop_mat_elt + exp(cmplx(0.0,1.0)*E_nk*t1)*x_nk(i)*x_km(i)*exp(cmplx(0.0,1.0)*E_km*t2)
					else if(key=='qp') then
						quadop_mat_elt = quadop_mat_elt + cmplx(0.0,1.0)*mass*exp(cmplx(0.0,1.0)*E_nk*t1)&
											*x_nk(i)*x_km(i)*E_km*exp(cmplx(0.0,1.0)*E_km*t2)
					else if(key=='pq') then
						quadop_mat_elt = quadop_mat_elt + cmplx(0.0,1.0)*mass*exp(cmplx(0.0,1.0)*E_nk*t1)&
											*x_nk(i)*x_km(i)*E_nk*exp(cmplx(0.0,1.0)*E_km*t2)
					else if(key=='pp') then
						quadop_mat_elt = quadop_mat_elt - mass*mass*exp(cmplx(0.0,1.0)*E_nk*t1)*x_nk(i)&
											*x_km(i)*E_nk*E_km*exp(cmplx(0.0,1.0)*E_km*t2)
					else if(key=='cm') then
						quadop_mat_elt = quadop_mat_elt + mass*(x_nk(i)*x_km(i)*&
									(E_km*exp(cmplx(0.0,1.0)*E_nk*t1)*exp(cmplx(0.0,1.0)*E_km*t2) -&
									 E_nk*exp(cmplx(0.0,1.0)*E_nk*t2)*exp(cmplx(0.0,1.0)*E_km*t1) ))
					end if
				end do
			end if
		end subroutine quadop_matrix_elts

		subroutine quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t1,t2,key,arrkey,quadop_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: lenk, n
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t1,t2,mass
			integer :: i
			character*2, intent(in) :: key
			character*3, intent(in) :: arrkey
			!f2py complex, dimension(lenm),intent(in,out,copy) :: quadop_mat
			complex, dimension(lenm),intent(inout) :: quadop_mat
			!$OMP PARALLEL DO PRIVATE(i)
			do i=1,lenm
				if (arrkey=='ket') then
					call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr(i),t1,t2,key,quadop_mat(i))
				else if(arrkey=='bra') then 
					call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,m_arr(i),n,t1,t2,key,quadop_mat(i)) 
				end if
			end do
			!$OMP END PARALLEL DO
		end subroutine quadop_matrix_arr

		subroutine quadop_matrix_arr_t(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t_arr,lent,torder,key,quadop_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lent,torder
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: lenk, n,m
			real(kind=8), dimension(lent),intent(in) :: t_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: mass
			integer :: i
			character*2, intent(in) :: key
			!f2py complex, dimension(lent),intent(in,out,copy) :: quadop_mat
			complex, dimension(lent),intent(inout) :: quadop_mat
			!$OMP PARALLEL DO PRIVATE(i)
			do i=1,lent
				if (torder==1) then
					call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t_arr(i),0.0d0,key,quadop_mat(i))
				else
					call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,0.0d0,t_arr(i),key,quadop_mat(i))
				end if
			end do
			!$OMP END PARALLEL DO
		end subroutine quadop_matrix_arr_t

		subroutine corr_mc_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
				lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,key,c_mc_elt)
			! Subroutine to compute standard correlators which are functions of one time variable
			! Keys: 'xp1'
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: lenk, n
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t, mass
			integer :: i
			character*3, intent(in) :: key
			!f2py complex,intent(in,out,copy) :: c_mc_elt
			complex,intent(inout) :: c_mc_elt
			complex, dimension(lenm) :: O_nm, O_mn
			c_mc_elt = cmplx(0.0,0.0)
			O_nm = cmplx(0.0,0.0)
			O_mn = cmplx(0.0,0.0)
			if (key=='qq1') then
				call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,n,t,0.0d0,'qq',c_mc_elt)
			else if (key=='qp1') then
				call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,n,t,0.0d0,'qp',c_mc_elt)
			else if (key=='pq1') then
				call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,n,t,0.0d0,'pq',c_mc_elt)
			else if (key=='pp1') then
				call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,n,t,0.0d0,'pp',c_mc_elt)
			else if (key=='xxC') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,0.0d0,'cm','ket',O_nm)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*conjg(O_nm(i))
				end do
			else if (key=='xxF') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,0.0d0,'qp','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,0.0d0,'qp','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='xG1') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,0.0d0,'qp','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,t,'pq','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='xG2') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,t,'pq','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,0.0d0,'qp','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='qq2') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,t,'qq','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,0.0d0,'qq','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='qp2') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,t,'qq','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,0.0d0,'pp','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='pq2') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,t,'pp','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,0.0d0,'qq','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			else if (key=='pp2') then
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,t,'pp','ket',O_nm)
				call quadop_matrix_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,0.0d0,0.0d0,'pp','bra',O_mn)
				do i=1,lenm
					c_mc_elt = c_mc_elt + O_nm(i)*O_mn(i)
				end do
			end if
		end subroutine corr_mc_elts

		subroutine corr_mc_arr(vecs,len1vecs,len2vecs,mass,x_arr,&
				lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n_arr,lenn,m_arr,lenm,t,key,c_mc_arr)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm,lenn
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, intent(in) :: lenk
			integer, dimension(lenn), intent(in) :: n_arr
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t,mass
			integer :: i
			character*3, intent(in) :: key
			!f2py complex,dimension(lenn),intent(in,out,copy) :: c_mc_arr
			complex,dimension(lenn),intent(inout) :: c_mc_arr
			!$OMP PARALLEL DO PRIVATE(i)
			do i=1,lenn
				call corr_mc_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n_arr(i),m_arr,lenm,t,key,c_mc_arr(i))
			end do
			!$OMP END PARALLEL DO
		end subroutine corr_mc_arr
		
		subroutine corr_mc_arr_t(vecs,len1vecs,len2vecs,mass,x_arr,&
				lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t_arr,lent,key,c_mc_arr)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm,lent,n
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy,mass
			integer, intent(in) :: lenk
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), dimension(lent), intent(in) :: t_arr
			integer :: i
			character*3, intent(in) :: key
			!f2py complex,dimension(lent),intent(in,out,copy) :: c_mc_arr
			complex,dimension(lent),intent(inout) :: c_mc_arr
			!$OMP PARALLEL DO PRIVATE(i)
			do i=1,lent
				call corr_mc_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t_arr(i),key,c_mc_arr(i))
			end do
			!$OMP END PARALLEL DO
		end subroutine corr_mc_arr_t

		subroutine therm_corr_elts(vecs,len1vecs,len2vecs,mass,x_arr,lenx,dx,dy,k_arr,lenk, &
					vals_arr,lenv,m_arr,lenm,t,beta,n_eigen,key,corr_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,n_eigen
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t,beta, mass
			integer :: i
			character*3, intent(in) :: key
			!f2py complex,intent(in,out,copy) :: corr_elt
			complex,intent(inout) :: corr_elt
			real :: Z
			complex, dimension(n_eigen) :: c_mc_mat
			c_mc_mat=cmplx(0.0,0.0)
			Z=0.0d0
			do i=1,n_eigen
				Z = Z + exp(-beta*vals_arr(i))
			end do
			do i=1,n_eigen
				call corr_mc_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
						lenx,dx,dy,k_arr,lenk,vals_arr,lenv,i,m_arr,lenm,t,key,c_mc_mat(i))
				corr_elt = corr_elt + (1/Z)*exp(-beta*vals_arr(i))*c_mc_mat(i)
			end do
		end subroutine therm_corr_elts

		subroutine kubo_corr_elts(vecs,len1vecs,len2vecs,mass,x_arr,lenx,dx,dy,k_arr,lenk, &
					vals_arr,lenv,m_arr,lenm,t,beta,n_eigen,key,corr_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,n_eigen
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t,beta, mass
			integer :: n,m
			character*3, intent(in) :: key
			!f2py complex,intent(in,out,copy) :: corr_elt
			complex,intent(inout) :: corr_elt
			real :: Z
			complex :: O_nm, O_mn 
			Z=0.0
			do n=1,n_eigen
				Z = Z + exp(-beta*vals_arr(n))
			end do
			do n=1,n_eigen
				do m=1,n_eigen
					O_nm = 0.0
					O_mn = 0.0
					if (key=='qq1' .or. key=='qp1' .or. key=='pq1' .or. key=='pp1') then 
						call linop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
							lenx,dx,dy,vals_arr,lenv,n,m,t,key(1:1),O_nm)
						call linop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
							lenx,dx,dy,vals_arr,lenv,n,m,0.0d0,key(2:2),O_mn)
					else if (key=='xxC') then
						call quadop_matrix_elts(vecs,len1vecs,len2vecs,mass,x_arr,&
							lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t,0.0d0,'cm',O_nm)
						O_mn = conjg(O_nm)
					end if
					!print*, O_nm
					if(n .NE. m) then
						corr_elt = corr_elt + (1/(Z*beta))*((exp(-beta*vals_arr(n)) - exp(-beta*vals_arr(m)))&
								/(vals_arr(m)-vals_arr(n)))*O_nm*O_mn
					else
						corr_elt = corr_elt + (1/Z)*exp(-beta*vals_arr(m))*O_nm*O_mn
					end if
				end do
			end do
		end subroutine kubo_corr_elts

		subroutine therm_corr_arr_t(vecs,len1vecs,len2vecs,mass,x_arr,lenx,dx,dy,k_arr,lenk, &
					vals_arr,lenv,m_arr,lenm,t_arr,lent,beta,n_eigen,key,corrkey,corr_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,lent,n_eigen
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lent), intent(in) :: t_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: beta, mass
			integer :: i
			character*3, intent(in) :: key
			character*4, intent(in) :: corrkey
			!f2py complex,dimension(lent),intent(in,out,copy) :: corr_mat
			complex,dimension(lent),intent(inout) :: corr_mat

			if (corrkey=='stan') then
				!$OMP PARALLEL DO PRIVATE(i)
				do i=1,lent
					call therm_corr_elts(vecs,len1vecs,len2vecs,mass,x_arr,lenx,dx,dy,k_arr,lenk, &
						vals_arr,lenv,m_arr,lenm,t_arr(i),beta,n_eigen,key,corr_mat(i))
				end do
				!$OMP END PARALLEL DO
			else if (corrkey=='kubo') then
				!$OMP PARALLEL DO PRIVATE(i)
				do i=1,lent
					call Kubo_corr_elts(vecs,len1vecs,len2vecs,mass,x_arr,lenx,dx,dy,k_arr,lenk, &
						vals_arr,lenv,m_arr,lenm,t_arr(i),beta,n_eigen,key,corr_mat(i))
				end do
				!$OMP END PARALLEL DO
			end if
		end subroutine therm_corr_arr_t

end module otoc_tools
