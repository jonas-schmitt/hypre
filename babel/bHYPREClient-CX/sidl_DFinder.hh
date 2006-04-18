// 
// File:          sidl_DFinder.hh
// Symbol:        sidl.DFinder-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.DFinder
// 
// Copyright (c) 2000-2002, The Regents of the University of California.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.DFinder-v0.9.3.xml
// 

#ifndef included_sidl_DFinder_hh
#define included_sidl_DFinder_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace ucxx { 
  namespace sidl { 

    class DFinder;
  } // end namespace sidl
} // end namespace ucxx

// Some compilers need to define array template before the specializations
#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
namespace ucxx {
  namespace sidl {
    template<>
    class array< ::ucxx::sidl::DFinder >;
  }
} //closes ucxx Namespace
// 
// Forward declarations for method dependencies.
// 
namespace ucxx { 
  namespace sidl { 

    class BaseInterface;
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 

    class ClassInfo;
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx { 
  namespace sidl { 

    class DLL;
  } // end namespace sidl
} // end namespace ucxx

#ifndef included_sidl_ucxx_hh
#include "sidl_ucxx.hh"
#endif
#ifndef included_sidl_DFinder_IOR_h
#include "sidl_DFinder_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#ifndef included_sidl_Finder_hh
#include "sidl_Finder.hh"
#endif
#ifndef included_sidl_Resolve_hh
#include "sidl_Resolve.hh"
#endif
#ifndef included_sidl_Scope_hh
#include "sidl_Scope.hh"
#endif

namespace ucxx { 
  namespace sidl { 

    /**
     * Symbol "sidl.DFinder" (version 0.9.3)
     * 
     *  This class is the Default Finder.  If no Finder is set in class Loader,
     *  this finder is used.  It uses SCL files from the filesystem to
     *  resolve dynamic libraries.
     * 
     * The initial search path is taken from the SIDL_DLL_PATH
     * environment variable.
     */
    class DFinder: public virtual ::ucxx::sidl::BaseClass,
      public virtual ::ucxx::sidl::Finder {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

    /**
     * <p>
     * Add one to the intrinsic reference count in the underlying object.
     * Object in <code>sidl</code> have an intrinsic reference count.
     * Objects continue to exist as long as the reference count is
     * positive. Clients should call this method whenever they
     * create another ongoing reference to an object or interface.
     * </p>
     * <p>
     * This does not have a return value because there is no language
     * independent type that can refer to an interface or a
     * class.
     * </p>
     */
    inline void
    addRef() throw () 
    {

      if ( !d_weak_reference ) {
        ior_t* loc_self = _get_ior();
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_addRef))(loc_self );
        /*dispatch to ior*/
        /*unpack results and cleanup*/
      }
    }



    /**
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in <code>sidl</code> have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     */
    inline void
    deleteRef() throw () 
    {

      if ( !d_weak_reference ) {
        ior_t* loc_self = _get_ior();
        /*pack args to dispatch to ior*/
        (*(loc_self->d_epv->f_deleteRef))(loc_self );
        /*dispatch to ior*/
        /*unpack results and cleanup*/
        d_self = 0;
      }
    }



    /**
     * Return true if and only if <code>obj</code> refers to the same
     * object as this object.
     */
    bool
    isSame (
      /* in */::ucxx::sidl::BaseInterface iobj
    )
    throw () 
    ;



    /**
     * Check whether the object can support the specified interface or
     * class.  If the <code>sidl</code> type name in <code>name</code>
     * is supported, then a reference to that object is returned with the
     * reference count incremented.  The callee will be responsible for
     * calling <code>deleteRef</code> on the returned object.  If
     * the specified type is not supported, then a null reference is
     * returned.
     */
    ::ucxx::sidl::BaseInterface
    queryInt (
      /* in */const ::std::string& name
    )
    throw () 
    ;



    /**
     * Return whether this object is an instance of the specified type.
     * The string name must be the <code>sidl</code> type name.  This
     * routine will return <code>true</code> if and only if a cast to
     * the string type name would succeed.
     */
    bool
    isType (
      /* in */const ::std::string& name
    )
    throw () 
    ;



    /**
     * Return the meta-data about the class implementing this interface.
     */
    ::ucxx::sidl::ClassInfo
    getClassInfo() throw () 
    ;


    /**
     * Find a DLL containing the specified information for a sidl
     * class. This method searches through the files in set set path
     * looking for a shared library that contains the client-side or IOR
     * for a particular sidl class.
     * 
     * @param sidl_name  the fully qualified (long) name of the
     *                   class/interface to be found. Package names
     *                   are separated by period characters from each
     *                   other and the class/interface name.
     * @param target     to find a client-side binding, this is
     *                   normally the name of the language.
     *                   To find the implementation of a class
     *                   in order to make one, you should pass
     *                   the string "ior/impl" here.
     * @param lScope     this specifies whether the symbols should
     *                   be loaded into the global scope, a local
     *                   scope, or use the setting in the file.
     * @param lResolve   this specifies whether symbols should be
     *                   resolved as needed (LAZY), completely
     *                   resolved at load time (NOW), or use the
     *                   setting from the file.
     * @return a non-NULL object means the search was successful.
     *         The DLL has already been added.
     */
    ::ucxx::sidl::DLL
    findLibrary (
      /* in */const ::std::string& sidl_name,
      /* in */const ::std::string& target,
      /* in */::ucxx::sidl::Scope lScope,
      /* in */::ucxx::sidl::Resolve lResolve
    )
    throw () 
    ;



    /**
     * Set the search path, which is a semi-colon separated sequence of
     * URIs as described in class <code>DLL</code>.  This method will
     * invalidate any existing search path.
     */
    void
    setSearchPath (
      /* in */const ::std::string& path_name
    )
    throw () 
    ;



    /**
     * Return the current search path.  If the search path has not been
     * set, then the search path will be taken from environment variable
     * SIDL_DLL_PATH.
     */
    ::std::string
    getSearchPath() throw () 
    ;


    /**
     * Append the specified path fragment to the beginning of the
     * current search path.  If the search path has not yet been set
     * by a call to <code>setSearchPath</code>, then this fragment will
     * be appended to the path in environment variable SIDL_DLL_PATH.
     */
    void
    addSearchPath (
      /* in */const ::std::string& path_fragment
    )
    throw () 
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct sidl_DFinder__object ior_t;
    typedef struct sidl_DFinder__external ext_t;
    typedef struct sidl_DFinder__sepv sepv_t;

    // default constructor
    DFinder() { }

    // static constructor
    static ::ucxx::sidl::DFinder _create();

    // default destructor
    virtual ~DFinder () { }

    // copy constructor
    DFinder ( const DFinder& original );

    // assignment operator
    DFinder& operator= ( const DFinder& rhs );

    // conversion from ior to C++ class
    DFinder ( DFinder::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    DFinder ( DFinder::ior_t* ior, bool isWeak );

    ior_t* _get_ior() { return reinterpret_cast< ior_t*>(d_self); }

    const ior_t* _get_ior() const { return reinterpret_cast< ior_t*>(d_self); }

    void _set_ior( ior_t* ptr ) { d_self = reinterpret_cast< void*>(ptr); }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

    static inline const char * type_name() { return "sidl.DFinder";}
    virtual void* _cast(const char* type) const;

  protected:
      // Pointer to external (DLL loadable) symbols (shared among instances)
      static const ext_t * s_ext;

    public:
      static const ext_t * _get_ext() throw ( ::ucxx::sidl::NullIORException );

    }; // end class DFinder
  } // end namespace sidl
} // end namespace ucxx

namespace ucxx {
  namespace sidl {
    // traits specialization
    template<>
    struct array_traits< ::ucxx::sidl::DFinder > {
      typedef array< ::ucxx::sidl::DFinder > cxx_array_t;
      typedef ::ucxx::sidl::DFinder cxx_item_t;
      typedef struct sidl_DFinder__array ior_array_t;
      typedef sidl_interface__array ior_array_internal_t;
      typedef struct sidl_DFinder__object ior_item_t;
      typedef cxx_item_t value_type;
      typedef value_type reference;
      typedef value_type* pointer;
      typedef const value_type const_reference;
      typedef const value_type* const_pointer;
      typedef array_iter< array_traits< ::ucxx::sidl::DFinder > > iterator;
      typedef const_array_iter< array_traits< ::ucxx::sidl::DFinder > > 
        const_iterator;
    };

    // array specialization
    template<>
    class array< ::ucxx::sidl::DFinder >: public interface_array< array_traits< 
      ::ucxx::sidl::DFinder > > {
    public:
      typedef interface_array< array_traits< ::ucxx::sidl::DFinder > > Base;
      typedef array_traits< ::ucxx::sidl::DFinder >::cxx_array_t          
        cxx_array_t;
      typedef array_traits< ::ucxx::sidl::DFinder >::cxx_item_t           
        cxx_item_t;
      typedef array_traits< ::ucxx::sidl::DFinder >::ior_array_t          
        ior_array_t;
      typedef array_traits< ::ucxx::sidl::DFinder >::ior_array_internal_t 
        ior_array_internal_t;
      typedef array_traits< ::ucxx::sidl::DFinder >::ior_item_t           
        ior_item_t;

      /**
       * conversion from ior to C++ class
       * (constructor/casting operator)
       */
      array( struct sidl_DFinder__array* src = 0) : Base(src) {}

      /**
       * copy constructor
       */
      array( const array< ::ucxx::sidl::DFinder >&src) : Base(src) {}

      /**
       * assignment
       */
      array< ::ucxx::sidl::DFinder >&
      operator =( const array< ::ucxx::sidl::DFinder >&rhs ) { 
        if (d_array != rhs._get_baseior()) {
          if (d_array) deleteRef();
          d_array = const_cast<sidl__array *>(rhs._get_baseior());
          if (d_array) addRef();
        }
        return *this;
      }

    };
  }

} //closes ucxx Namespace
#endif