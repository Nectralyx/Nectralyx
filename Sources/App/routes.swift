import Vapor

func routes(_ app: Application) throws {
    app.get { req async in
        "It works!"
    }

    app.get("hello") { req async -> String in
        "Hello, world!"
    }
    
    app.get("test") { req async -> String in
"""
---OverDrive DataBase---
        
           /|
          /||
         //||
--------///||*******///-
-------///*||******///--
------///**||*****///---
-----///***||****///----
----///****||***///-----
---///*****||**///------
--///******||*///-------
-///*******||///--------
           ||//
           |//
           |/
"""
        
    }
    app.post("send") { req -> String in
       let message = try req.content.decode(Send.self)
        return "You sent: \(message.message)"
    }
    
    
}

