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

            /
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
           /
"""
        
    }
        struct  Send: Content {
            let message: String
        }
    // 2
    
    app.post("send") { req async -> String in
        
        let data = try? req.content.decode(Send.self)
        
        return "Hello \(data?.message ?? "String")!"
    }
    
    app.get("update") { req async -> String in
        "Up to date"
    }
}
