


module position_matrix	
	implicit none
	private
	public :: pos_matrix_elts
	public :: testing
	public :: compute_pos_mat_arr_k
	public :: compute_pos_mat_arr_n
	public :: b_matrix_elts
	public :: compute_b_mat_arr_m
	public :: c_mc_elts
	public :: compute_c_mc_arr_n
	public :: compute_c_mc_arr_t
	public :: OTOC_elts
	public :: compute_OTOC_arr_t
	public :: Kubo_OTOC_elts
	public :: compute_Kubo_OTOC_arr_t
	contains
		subroutine testing()
			print*, "JUst cheking"
		end subroutine testing

		subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
			integer, intent(in) :: n,k , len1vecs,len2vecs,lenx			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			!f2py real(kind=8), intent(in,out,copy) :: pos_mat_elt
			integer :: i,j
			real(kind=8), intent(inout) :: pos_mat_elt
			!print*, "Here", shape(vecs),len1vecs
			!print*, x_arr(10),x_arr(14),x_arr(20)
			!print*, vecs(21,13),vecs(18,12),vecs(14,9)
			do i = 1,len1vecs
					pos_mat_elt = pos_mat_elt + vecs(i,n+1)*vecs(i,k+1)*x_arr(i)*dx*dy !! Change here for 1D 
			end do

		end subroutine pos_matrix_elts

		subroutine compute_pos_mat_arr_k(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr,lenk,pos_mat)
			integer, intent(in) ::  len1vecs,len2vecs,lenx,lenk,n
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs	
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer :: i,j
			!f2py real(kind=8), dimension(lenk), intent(in,out,copy) :: pos_mat
			real(kind=8), dimension(lenk), intent(inout) :: pos_mat
			do i=1,lenk
				call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr(i),pos_mat(i))	
			end do
		end subroutine compute_pos_mat_arr_k

		subroutine compute_pos_mat_arr_n(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n_arr,lenn,k,pos_mat)
			integer, intent(in) ::  len1vecs,len2vecs,lenx,lenn,k
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs	
			integer, dimension(lenn), intent(in) :: n_arr
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
			integer :: i,j
			!f2py real(kind=8), dimension(lenn), intent(in,out,copy) :: pos_mat
			real(kind=8), dimension(lenn), intent(inout) :: pos_mat
			do i=1,lenn
				call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n_arr(i),k,pos_mat(i))	
			end do
		end subroutine compute_pos_mat_arr_n

		subroutine b_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t,b_mat_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, intent(in) :: lenk, n, m
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t
			integer :: i
			!f2py complex, intent(in,out,copy) :: b_mat_elt
			complex(kind=8), intent(inout) :: b_mat_elt
			real(kind=8) :: E_km, E_nk
			real(kind=8), dimension(lenk) :: x_nk, x_km
			x_nk = 0.0
			x_km = 0.0
			call compute_pos_mat_arr_k(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k_arr,lenk,x_nk)
			call compute_pos_mat_arr_n(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,m,x_km)
			do i=1,lenk
				E_km = vals_arr(k_arr(i)+1) - vals_arr(m+1)
				E_nk = vals_arr(n+1) - vals_arr(k_arr(i)+1)
				b_mat_elt = b_mat_elt + 0.5*(x_nk(i)*x_km(i)*(E_km*exp(cmplx(0.0,1.0)*E_nk*t) - E_nk*exp(cmplx(0.0,1.0)*E_km*t)))
			end do
		end subroutine b_matrix_elts

		subroutine compute_b_mat_arr_m(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,b_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, intent(in) :: lenk, n
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t
			integer :: i
			!f2py complex, dimension(lenm),intent(in,out,copy) :: b_mat
			complex(kind=8), dimension(lenm),intent(inout) :: b_mat
			!call b_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr(2),t,b_mat(2))
			do i=1,lenm
				call b_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr(i),t,b_mat(i))
			end do
		end subroutine compute_b_mat_arr_m


		subroutine c_mc_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,c_mc_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, intent(in) :: lenk, n
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t
			integer :: i
			!f2py real,intent(in,out,copy) :: c_mc_elt
			real,intent(inout) :: c_mc_elt
			complex(kind=8), dimension(lenm) :: b_nm
			b_nm=0.0
			call compute_b_mat_arr_m(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t,b_nm)
			do i=1,lenm
				c_mc_elt = c_mc_elt + b_nm(i)*conjg(b_nm(i))
			end do
		
		end subroutine c_mc_elts

		subroutine compute_c_mc_arr_n(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n_arr,lenn,m_arr,lenm,t,c_mc_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm,lenn			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, intent(in) :: lenk
			integer, dimension(lenn), intent(in) :: n_arr
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t
			integer :: i
			!f2py real,dimension(lenn),intent(in,out,copy) :: c_mc_mat
			real,dimension(lenn),intent(inout) :: c_mc_mat

			do i=1,lenn
				call c_mc_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n_arr(i),m_arr,lenm,t,c_mc_mat(i))
			end do
		end subroutine compute_c_mc_arr_n

		subroutine compute_c_mc_arr_t(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t_arr,lent,c_mc_mat)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenm,lent,n			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, intent(in) :: lenk
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), dimension(lent), intent(in) :: t_arr
			integer :: i
			!f2py real,dimension(lent),intent(in,out,copy) :: c_mc_mat
			real,dimension(lent),intent(inout) :: c_mc_mat
			do i=1,lent
				call c_mc_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m_arr,lenm,t_arr(i),c_mc_mat(i))
			end do
		end subroutine compute_c_mc_arr_t

		subroutine OTOC_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,m_arr,lenm,t,beta,n_eigen,OTOC_elt)
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,n_eigen			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t,beta
			integer :: i
			!f2py real,intent(in,out,copy) :: OTOC_elt
			real,intent(inout) :: OTOC_elt
			real :: Z
			real, dimension(n_eigen) :: c_mc_mat
			c_mc_mat=0.0
			Z=0.0
			do i=1,n_eigen
				Z = Z + exp(-beta*vals_arr(i))
			end do
			do i=1,n_eigen
				call c_mc_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,i-1,m_arr,lenm,t,c_mc_mat(i))
				OTOC_elt = OTOC_elt + (1/Z)*exp(-beta*vals_arr(i))*c_mc_mat(i)
			end do
				
		end subroutine OTOC_elts

		subroutine compute_OTOC_arr_t(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,m_arr,lenm,&
t_arr,lent,beta,n_eigen,OTOC_mat)	
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,lent,n_eigen			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lent), intent(in) :: t_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: beta
			integer :: i
			!f2py real,dimension(lent),intent(in,out,copy) :: OTOC_mat
			real,dimension(lent),intent(inout) :: OTOC_mat
			do i=1,lent
				call OTOC_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,m_arr,lenm,t_arr(i),beta,n_eigen,OTOC_mat(i))
			end do
		end subroutine compute_OTOC_arr_t

		subroutine Kubo_OTOC_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,m_arr,lenm,t,beta,n_eigen,OTOC_elt)

			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,n_eigen			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: t,beta
			integer :: n,m
			!f2py real,intent(in,out,copy) :: OTOC_elt
			real,intent(inout) :: OTOC_elt
			real(kind=8) :: Z
			complex(kind=8) :: b_nm
			real, dimension(n_eigen) :: c_mc_mat
			c_mc_mat=0.0
			Z=0.0
			do n=1,n_eigen
				Z = Z + exp(-beta*vals_arr(n))
			end do
			do n=1,n_eigen
				do m=1,n_eigen
					b_nm = 0.0
					call b_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv,n,m,t,b_nm)
					!OTOC_elt = OTOC_elt + &
					!		(1/(beta*Z*(vals_arr(m)-vals_arr(n))))*(exp(-beta*(vals_arr(n) &
					!		-vals_arr(m)))-1)*exp(-beta*vals_arr(m))*b_nm*conjg(b_nm)
					if(n .NE. m) then
					OTOC_elt = OTOC_elt + (1/(Z*beta))*((exp(-beta*vals_arr(n)) - exp(-beta*vals_arr(m)))&
							/(vals_arr(m)-vals_arr(n)))*b_nm*conjg(b_nm)
					else
					OTOC_elt = OTOC_elt + (1/Z)*exp(-beta*vals_arr(m))*b_nm*conjg(b_nm)
					end if
				end do
			end do
				
		end subroutine Kubo_OTOC_elts

		subroutine compute_Kubo_OTOC_arr_t(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv, &
				m_arr,lenm,t_arr,lent,beta,n_eigen,Kubo_OTOC_mat)	
			integer, intent(in) :: len1vecs,len2vecs,lenx,lenv,lenk,lenm,lent,n_eigen			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy			
			integer, dimension(lenm),intent(in) :: m_arr
			integer, dimension(lenk), intent(in) :: k_arr
			real(kind=8), dimension(lent), intent(in) :: t_arr
			real(kind=8), dimension(lenv), intent(in) :: vals_arr
			real(kind=8), intent(in) :: beta
			integer :: i
			!f2py real,dimension(lent),intent(in,out,copy) :: Kubo_OTOC_mat
			real,dimension(lent),intent(inout) :: Kubo_OTOC_mat
			!print*, 'Here'
			do i=1,lent
				call Kubo_OTOC_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,k_arr,lenk,vals_arr,lenv&
					,m_arr,lenm,t_arr(i),beta,n_eigen,Kubo_OTOC_mat(i))
			end do
		end subroutine compute_Kubo_OTOC_arr_t
end module position_matrix
