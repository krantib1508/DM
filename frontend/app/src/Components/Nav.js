import React, { useEffect } from 'react'
import './Nav.css'

const Nav = () => {
  return (
    <div>
    <div class="page-wrapper chiller-theme toggled">
  <a id="show-sidebar" class="btn btn-sm btn-dark" href="#">
    <i class="fas fa-bars"></i>
  </a>
  
  <nav id="sidebar" class="sidebar-wrapper">
    <div class="sidebar-content">
      <div class="sidebar-brand">
        <div id="close-sidebar"><i class="fas fa-times"></i></div>
      </div>
      <div class="sidebar-header">
        <div class="user-pic" style={{color:"color:#fff"}}>
          <i class="fa fa-user-circle fa-4x" aria-hidden="true"></i>
        </div>
        <div class="user-info">
          <span class="user-name"> <strong>Kranti Bharti</strong></span>
          <span class="user-role"> </span>
        </div>
      </div>
      
      <div class="sidebar-menu">
        <ul>
          <li class="sidebar-dropdown">
            <a href="/file_upload"><i class="fa fa-tachometer-alt"></i><span>File Upload</span><span class="badge badge-pill badge-warning"></span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_1"><i class="fa fa-shopping-cart"></i><span>Assignment 1</span><span class="badge badge-pill badge-danger"></span></a>
            <div class="sidebar-submenu">
             
            </div>
          </li>

          <li class="sidebar-dropdown">
            <a href="/assignment_1_ques2"><i class="fa fa-shopping-cart"></i><span>Assignment 1 plots</span><span class="badge badge-pill badge-danger"></span></a>
            <div class="sidebar-submenu">
             
            </div>
          </li>



          <li class="sidebar-dropdown">
            <a href="/assignment_2"><i class="far fa-gem"></i><span>Assignment 2</span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_3"><i class="fa fa-chart-line"></i><span>Assignment 3</span></a>
            <div class="sidebar-submenu">
           
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_4"><i class="fa fa-globe"></i><span>Assignment 4</span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>

          <li class="sidebar-dropdown">
            <a href="/assignment_5"><i class="fa fa-globe"></i><span>Assignment 5</span></a>
            <div class="sidebar-submenu">
             
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_6"><i class="fa fa-globe"></i><span>Assignment 6</span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_7"><i class="fa fa-globe"></i><span>Assignment 7</span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>
          <li class="sidebar-dropdown">
            <a href="/assignment_8"><i class="fa fa-globe"></i><span>Assignment 8</span></a>
            <div class="sidebar-submenu">
            
            </div>
          </li>
          
          
        
        </ul>
      </div>
    </div>
    
  </nav>
    
  <main class="page-content">
    <div class="container-fluid">
      <hr/>
    </div>
  </main>
</div>
    
     </div>
  )
}

export default Nav