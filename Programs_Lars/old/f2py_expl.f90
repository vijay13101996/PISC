!compile using f2py -c f2py_expl.f90 -m fortran
module foo
    implicit none
contains 
    subroutine existence()
        print *,'Im ALIVE!'
    end subroutine existence
    subroutine fast_reverse(a,n)
        !reverses first n el of array
        integer, intent(in) :: n 
        real, intent(inout) :: a(:) !careful with type in python

        a(1:n) = a(n:1:-1)
    end subroutine fast_reverse
end module foo

!Bem ggf mit pyf compilen aber sollte passen

!!!fortran.pyf
!compile using f2py f2py_expl.f90 -m fortran -h fortran.pyf
!python module fortran
!    interface 
!        module fmodule
!            subroutine fast_reverse(a,n)
!                real dimension(:), intent(inout) :: a 
!                integer intent(in) :: n 
!            end subroutine fast_reverse
!        end module fmodule
!    end interface
!end python module fortran

