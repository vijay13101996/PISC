module husimi_section
	use omp_lib
	implicit none
	private
	public :: coherent_state
	public :: coherent_projection
	public :: husimi_section_x
	public :: husimi_section_y
	public :: husimi_rep_x
	public :: husimi_rep_y
	public :: husimi_rep
	public :: renyi_entropy
	public :: renyi_entropy_1d
	real, parameter :: pi = 3.1415927
	contains
		subroutine coherent_state(x,y,lengridx, lengridy, x0,px0,y0,py0,sigmax,sigmay,hbar,coh)
			integer, intent(in) :: lengridx, lengridy
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			real(kind=8), intent(in) :: x0,px0,y0,py0
			real(kind=8), intent(in) :: hbar, sigmax, sigmay
			complex :: cohx, cohy 
			!f2py complex, dimension(lengridx, lengridy), intent(in,out,copy) :: coh
			complex(kind=8), dimension(lengridx, lengridy), intent(inout) :: coh
			integer :: i,j
			!$OMP PARALLEL DO PRIVATE(i,j,cohx,cohy)
			do i = 1,lengridx
				do j = 1,lengridy
					cohx = (1/(pi*hbar*sigmax**2))**0.25*exp(-(x(i,j)-x0)**2/(2*hbar*sigmax**2) + cmplx(0.0,1.0)*px0*(x(i,j)-x0)/hbar) !!-x0
					cohy = (1/(pi*hbar*sigmay**2))**0.25*exp(-(y(i,j)-y0)**2/(2*hbar*sigmay**2) + cmplx(0.0,1.0)*py0*(y(i,j)-y0)/hbar) !!-y0
					coh(i,j) = cohx*cohy
				end do
			end do
			!$OMP END PARALLEL DO
		end subroutine coherent_state

		subroutine coherent_projection(x,y,lengridx, lengridy, x0,px0,y0,py0,sigmax,sigmay,dx,dy,hbar,wf,proj)
			integer, intent(in) :: lengridx, lengridy
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: x0,px0,y0,py0
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy
			!f2py complex,  intent(in,out,copy) :: proj
			complex(kind=8), intent(inout) :: proj
			complex(kind=8), dimension(lengridx,lengridy) :: coh
			integer :: i,j
			call coherent_state(x,y,lengridx,lengridy, x0,px0,y0,py0, sigmax,sigmay,hbar,coh)
			proj = 0.0
			!!$OMP PARALLEL DO PRIVATE(i,j)
			do i = 1,lengridx
				do j = 1,lengridy
					proj = proj + conjg(coh(i,j))*wf(i,j)*dx*dy
				end do
			end do
			!!$OMP END PARALLEL DO
		end subroutine coherent_projection

		subroutine husimi_section_x(x,y,lengridx,lengridy,xbasis,lenx,pxbasis,lenpx,&
					y0,potgrid,wf,E_wf,m,hbar,sigmax, sigmay, dx, dy,dist)
			integer, intent(in) :: lengridx, lengridy,lenx,lenpx
			real(kind=8), dimension(lenx), intent(in) :: xbasis
			real(kind=8), dimension(lenpx), intent(in) :: pxbasis
			real(kind=8), dimension(lenx), intent(in) :: potgrid
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy, E_wf,m,y0
			complex(kind=8), dimension(lengridx,lengridy) :: coh
			!f2py real(kind=8), dimension(lenx,lenpx), intent(in,out,copy) :: dist
			real(kind=8), dimension(lenx, lenpx), intent(inout) :: dist
			real(kind=8) :: pot, py0_sq, py0
			complex(kind=8) :: proj
			integer :: i,j

			dist = 0.0
			!!$OMP PARALLEL DO PRIVATE(i,j,py0,py0_sq,pot)
			do i=1,lenx
				do j=1,lenpx
					pot = potgrid(i)
					py0_sq = (2*m*(E_wf - pot - pxbasis(j)**2/(2*m)))
					if (py0_sq > 0.0) then
						py0 = py0_sq**0.5
						!if ( (abs(xbasis(i))<0.2) .and. (abs(pxbasis(j))<0.2) ) then
						!	print*, xbasis(i), pxbasis(j),'py0', py0
						!end if
						!call coherent_projection(x,y,lengridx,lengridy,&
						!	xbasis(i),pxbasis(j),ybasis(k),py0,sigmax,sigmay,dx,dy,hbar,wf, proj) 
						!dist(i,j) = dist(i,j) + abs(proj)**2
						call coherent_projection(x,y,lengridx,lengridy,&
							xbasis(i),pxbasis(j),y0,-py0,sigmax,sigmay,dx,dy,hbar,wf, proj) 
						!if (abs(proj)**2 > 2e-1) then
						!	print*, xbasis(i), pxbasis(j),abs(proj)**2
						!end if
						dist(i,j) = dist(i,j) + abs(proj)**2 
					!print*, 'dij mod after k', i,j,k,dist(i,j)
					end if
				end do
			end do
			!!$OMP END PARALLEL DO
		end subroutine husimi_section_x

		subroutine husimi_section_y(x,y,lengridx,lengridy,ybasis,leny,pybasis,lenpy,&
					x0,potgrid,wf,E_wf,m,hbar,sigmax, sigmay, dx, dy,dist)
			integer, intent(in) :: lengridx, lengridy,lenpy,leny
			real(kind=8), dimension(leny), intent(in) :: ybasis
			real(kind=8), dimension(lenpy), intent(in) :: pybasis
			real(kind=8), dimension(leny), intent(in) :: potgrid
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy, E_wf,m,x0
			complex(kind=8), dimension(lengridx,lengridy) :: coh
			!f2py real(kind=8), dimension(leny,lenpy), intent(in,out,copy) :: dist
			real(kind=8), dimension(leny, lenpy), intent(inout) :: dist
			real(kind=8) :: pot, px0_sq, px0
			complex(kind=8) :: proj
			integer :: i,j

			dist = 0.0
			!!$OMP PARALLEL DO PRIVATE(i,j,px0,px0_sq,pot)
			do i=1,leny
				do j=1,lenpy
					pot = potgrid(i)
					px0_sq = (2*m*(E_wf - pot - pybasis(j)**2/(2*m)))
					if (px0_sq > 0.0) then
						px0 = px0_sq**0.5
						call coherent_projection(x,y,lengridx,lengridy,&
							x0,px0,ybasis(i),pybasis(j),sigmax,sigmay,dx,dy,hbar,wf, proj) 
						dist(i,j) = dist(i,j) + abs(proj)**2
						!call coherent_projection(x,y,lengridx,lengridy,&
						!	xbasis(k),-px0,ybasis(i),pybasis(j),sigmax,sigmay,dx,dy,hbar,wf, proj) 
						!dist(i,j) = dist(i,j) + abs(proj)**2
					!print*, 'dij mod after k', i,j,k,dist(i,j)
					end if
				end do
			end do
			!!$OMP END PARALLEL DO
		end subroutine husimi_section_y

		subroutine husimi_rep_x(x,y,lengridx,lengridy,xbasis,lenx,pxbasis,lenpx,&
					wf,E_wf,m,hbar,sigmax, sigmay, dx, dy,dist)
			integer, intent(in) :: lengridx, lengridy,lenx,lenpx
			real(kind=8), dimension(lenx), intent(in) :: xbasis
			real(kind=8), dimension(lenpx), intent(in) :: pxbasis
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy, E_wf,m
			complex(kind=8), dimension(lengridx,lengridy) :: coh
			!f2py real(kind=8), dimension(lenx,lenpx), intent(in,out,copy) :: dist
			real(kind=8), dimension(lenx, lenpx), intent(inout) :: dist
			complex(kind=8) :: proj
			integer :: i,j

			dist = 0.0
			!!$OMP PARALLEL DO PRIVATE(i,j,k,py0,py0_sq,pot) SHARED(dist)
			do i=1,lenx
				do j=1,lenpx
					call coherent_projection(x,y,lengridx,lengridy,&
								xbasis(i),pxbasis(j),0.0d0,0.0d0,sigmax,sigmay,dx,dy,hbar,wf, proj) 
							dist(i,j) = dist(i,j) + abs(proj)**2
				end do
			end do
			!!$OMP END PARALLEL DO
		end subroutine husimi_rep_x

		subroutine husimi_rep_y(x,y,lengridx,lengridy,ybasis,leny,pybasis,lenpy,&
					wf,E_wf,m,hbar,sigmax, sigmay, dx, dy,dist)
			integer, intent(in) :: lengridx, lengridy,leny,lenpy
			real(kind=8), dimension(leny), intent(in) :: ybasis
			real(kind=8), dimension(lenpy), intent(in) :: pybasis
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy, E_wf,m
			complex(kind=8), dimension(lengridx,lengridy) :: coh
			!f2py real(kind=8), dimension(leny,lenpy), intent(in,out,copy) :: dist
			real(kind=8), dimension(leny, lenpy), intent(inout) :: dist
			complex(kind=8) :: proj
			integer :: i,j

			dist = 0.0
			!!$OMP PARALLEL DO PRIVATE(i,j,k,py0,py0_sq,pot) SHARED(dist)
			do i=1,leny
				do j=1,lenpy
					call coherent_projection(x,y,lengridx,lengridy,&
								0.0d0,0.0d0,ybasis(i),pybasis(j),sigmax,sigmay,dx,dy,hbar,wf, proj) 
							dist(i,j) = dist(i,j) + abs(proj)**2
				end do
			end do
			!!$OMP END PARALLEL DO
		end subroutine husimi_rep_y

		subroutine husimi_rep(x,y,lengridx,lengridy,xbasis,lenx,pxbasis,lenpx,ybasis,&
					leny,pybasis,lenpy,wf,E_wf,m,hbar,sigmax,sigmay,dx,dy,dist)
			integer, intent(in) :: lengridx, lengridy,lenx,lenpx,leny,lenpy
			real(kind=8), dimension(lenx), intent(in) :: xbasis
			real(kind=8), dimension(lenpx), intent(in) :: pxbasis
			real(kind=8), dimension(leny), intent(in) :: ybasis
			real(kind=8), dimension(lenpy), intent(in) :: pybasis
			real(kind=8), dimension(lengridx,lengridy), intent(in) :: x,y
			complex(kind=8), dimension(lengridx,lengridy), intent(in) :: wf
			real(kind=8), intent(in) :: hbar, sigmax, sigmay, dx, dy, E_wf,m
			!f2py complex(kind=8), dimension(lenx,lenpx,leny,lenpy), intent(in,out,copy) :: dist
			complex(kind=8), dimension(lenx,lenpx,leny, lenpy), intent(inout) :: dist
			complex(kind=8) :: proj
			integer :: i,j,k,l

			dist = 0.0
			!$OMP PARALLEL DO PRIVATE(i,j,k,l) 
			do i=1,lenx
				do j=1,lenpx
					do k=1,leny
						do l=1,lenpy
							call coherent_projection(x,y,lengridx,lengridy,xbasis(i),pxbasis(j),&
								ybasis(k),pybasis(l),sigmax,sigmay,dx,dy,hbar,wf, proj) 
							dist(i,j,k,l) = abs(proj)**2
						end do
					end do
				end do
			end do
			!$OMP END PARALLEL DO
		end subroutine husimi_rep

		subroutine renyi_entropy(xbasis,lenx,pxbasis,lenpx,ybasis,&
					leny,pybasis,lenpy,dist,S)
			integer, intent(in) :: lenx,lenpx,leny,lenpy
			real(kind=8), dimension(lenx), intent(in) :: xbasis
			real(kind=8), dimension(lenpx), intent(in) :: pxbasis
			real(kind=8), dimension(leny), intent(in) :: ybasis
			real(kind=8), dimension(lenpy), intent(in) :: pybasis
			real(kind=8), dimension(lenx,lenpx,leny, lenpy), intent(in) :: dist
			!f2py real(kind=8), intent(in,out,copy) :: S
			real(kind=8), intent(inout) :: S
			integer :: i,j,k,l
			real(kind=8) :: dx,dpx,dy,dpy
			S =0.0
			dx = xbasis(2)-xbasis(1)
			dy = ybasis(2)-ybasis(1)
			dpx = pxbasis(2)-pxbasis(1)
			dpy = pybasis(2)-pybasis(1)

			do i=1,lenx
				do j=1,lenpx
					do k=1,leny
						do l=1,lenpy
							if (dist(i,j,k,l) > 0.0) then
								S = S + dist(i,j,k,l)*log(dist(i,j,k,l))*dx*dy*dpx*dpy
							end if
						end do
					end do
				end do
			end do
			
		end subroutine renyi_entropy

		subroutine renyi_entropy_1d(qbasis,lenq,pbasis,lenp,dist,order,S)
			integer, intent(in) :: lenq,lenp,order
			real(kind=8), dimension(lenq), intent(in) :: qbasis
			real(kind=8), dimension(lenp), intent(in) :: pbasis
			real(kind=8), dimension(lenq,lenp), intent(in) :: dist
			!f2py real(kind=8), intent(in,out,copy) :: S
			real(kind=8), intent(inout) :: S
			integer :: i,j
			real(kind=8) :: dq,dp
			S =0.0
			dq = qbasis(2)-qbasis(1)
			dp = pbasis(2)-pbasis(1)

			do i=1,lenq
				do j=1,lenp
					if (dist(i,j) > 0.0) then
						if(order==1) then
							S = S + dist(i,j)*log(dist(i,j))*dq*dp
						else
							S = S + dist(i,j)**order*dq*dp
						end if
					end if
				end do
			end do
			
		end subroutine renyi_entropy_1d

end module husimi_section

